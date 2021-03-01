use std::future::Future;
use std::collections::HashMap;
use std::sync::Arc;
use num::rational::Rational64;
use num::ToPrimitive;
use ndarray::{Axis, Array, ArcArray, Ix2};
use ndarray_ops::MapArray3by3;
use godunov_core::runge_kutta;
use kepler_two_body::OrbitalElements;
use crate::physics::{
    Direction,
    CellData,
    HydroError,
    ItemizedChange,
    Solver,
};
use crate::mesh::{
    Mesh,
    BlockIndex,
};
use crate::traits::{
    Hydrodynamics,
    Conserved,
    ItemizeData,
};

use crate::tracers::{
    Tracer,
    update_tracer,
    push_new_tracers,
    tracers_on_and_off_block,
};




// ============================================================================
#[derive(Clone)]
pub struct BlockData<C: Conserved> {
    pub initial_conserved: ArcArray<C, Ix2>,
    pub cell_centers:      ArcArray<(f64, f64), Ix2>,
    pub face_centers_x:    ArcArray<(f64, f64), Ix2>,
    pub face_centers_y:    ArcArray<(f64, f64), Ix2>,
    pub index:             BlockIndex,
}




// ============================================================================
#[derive(Clone)]
pub struct BlockSolution<C: Conserved> {
    pub conserved: ArcArray<C, Ix2>,
    pub integrated_source_terms: ItemizedChange<C>,
    pub orbital_elements_change: ItemizedChange<OrbitalElements>,
    pub tracers: Arc<Vec<Tracer>>,
}

impl<C: Conserved> BlockSolution<C>
{
    pub fn with_tracers(&self, new_tracers: Vec<Tracer>) -> Self
    {
        BlockSolution{
            conserved: self.conserved.clone(),
            integrated_source_terms: self.integrated_source_terms,
            orbital_elements_change: self.orbital_elements_change,
            tracers: Arc::new(new_tracers),
        }
    }
}




// ============================================================================
#[derive(Clone)]
struct BlockState<C: Conserved> {
    pub time: f64,
    pub iteration: Rational64,
    pub solution: BlockSolution<C>,
}




// ============================================================================
#[derive(Clone)]
pub struct State<C: Conserved> {
    pub time: f64,
    pub iteration: Rational64,
    pub solution: Vec<BlockSolution<C>>,
}

impl<C: Conserved> State<C> {
    pub fn with_solution(&self, solution: Vec<BlockSolution<C>>) -> Self {
        State {
            time: self.time,
            iteration: self.iteration,
            solution: solution,
        }
    }

    pub fn total_tracers(&self) -> usize {
        self.solution.iter().map(|s| s.tracers.len()).sum()
    }
}




// ============================================================================
#[derive(Copy, Clone)]
struct UpdateScheme<H: Hydrodynamics> {
    hydro: H,
}




// ============================================================================
impl<C: ItemizeData> runge_kutta::WeightedAverage for ItemizedChange<C> {
    fn weighted_average(self, br: Rational64, s0: &Self) -> Self {
        let bf = br.to_f64().unwrap();
        Self {
            sink1:   self.sink1   * (-bf + 1.) + s0.sink1   * bf,
            sink2:   self.sink2   * (-bf + 1.) + s0.sink2   * bf,
            grav1:   self.grav1   * (-bf + 1.) + s0.grav1   * bf,
            grav2:   self.grav2   * (-bf + 1.) + s0.grav2   * bf,
            buffer:  self.buffer  * (-bf + 1.) + s0.buffer  * bf,
            cooling: self.cooling * (-bf + 1.) + s0.cooling * bf,
            fake_mass: self.fake_mass * (-bf + 1.) + s0.fake_mass * bf,
        }
    }
}




// ============================================================================
impl<C: Conserved> runge_kutta::WeightedAverage for BlockState<C> {
    fn weighted_average(self, br: Rational64, s0: &Self) -> Self {
        let bf = br.to_f64().unwrap();
        Self {
            time:      self.time      * (-bf + 1.) + s0.time      * bf,
            iteration: self.iteration * (-br + 1 ) + s0.iteration * br,
            solution:  self.solution.weighted_average(br, &s0.solution),
        }
    }
}

impl<C: Conserved> runge_kutta::WeightedAverage for BlockSolution<C> {
    fn weighted_average(self, br: Rational64, s0: &Self) -> Self {
        let s1 = self;
        let bf = br.to_f64().unwrap();
        let u0 = s0.conserved.clone();
        let u1 = s1.conserved.clone();
        let t0 = &s0.integrated_source_terms;
        let t1 = &s1.integrated_source_terms;
        let e0 = &s0.orbital_elements_change;
        let e1 = &s1.orbital_elements_change;
        let tr0 = &s0.tracers;
        let tr1 = &s1.tracers;

        BlockSolution{
            conserved: u1 * (-bf + 1.) + u0 * bf,
            integrated_source_terms: t1.weighted_average(br, t0),
            orbital_elements_change: e1.weighted_average(br, e0),
            tracers: Arc::new(tr0.iter().zip(tr1.as_ref()).map(|(tr0, tr1)| tr1.weighted_average(br, tr0)).collect()),
        }
    }
}




// ============================================================================
#[async_trait::async_trait]
impl<C: Conserved> runge_kutta::WeightedAverageAsync for State<C> {
    type Runtime = tokio::runtime::Runtime;

    async fn weighted_average(self, br: Rational64, s0: &Self, runtime: &Self::Runtime) -> Self {
        use futures::future::join_all;
        use godunov_core::runge_kutta::WeightedAverage;

        let bf = br.to_f64().unwrap();
        let s_avg = self.solution
            .into_iter()
            .zip(&s0.solution)
            .map(|(s1, s0)| (s1, s0.clone()))
            .map(|(s1, s0)| runtime.spawn(async move { s1.weighted_average(br, &s0) }))
            .map(|f| async { f.await.unwrap() });

        State {
            time:      self.time      * (-bf + 1.) + s0.time      * bf,
            iteration: self.iteration * (-br + 1 ) + s0.iteration * br,
            solution: join_all(s_avg).await,
        }
    }
}




// ============================================================================
async fn try_join_3by3<F, T, E>(a: [[&F; 3]; 3]) -> Result<[[T; 3]; 3], E>
where
    F: Clone + Future<Output = Result<T, E>>
{
    Ok([
        [a[0][0].clone().await?, a[0][1].clone().await?, a[0][2].clone().await?],
        [a[1][0].clone().await?, a[1][1].clone().await?, a[1][2].clone().await?],
        [a[2][0].clone().await?, a[2][1].clone().await?, a[2][2].clone().await?],
    ])
}




// ============================================================================
async fn try_advance_tokio_rk<H: 'static + Hydrodynamics>(
    state: State<H::Conserved>,
    hydro: H,
    block_data: &Vec<BlockData<H::Conserved>>,
    mesh: &Mesh,
    solver: &Solver,
    dt: f64,
    runtime: &tokio::runtime::Runtime) -> Result<State<H::Conserved>, HydroError>
{
    use futures::future::{FutureExt, join_all};

    let scheme = UpdateScheme::new(hydro);
    let time = state.time;
    let mut pc_map = HashMap::new();
    let mut fv_map = HashMap::new();
    let mut s1_vec = Vec::new();

    for (solution, block) in state.solution.iter().zip(block_data) {
        let uc = solution.conserved.clone();
        let block = block.clone();
        let block_index = block.index;
        let primitive = async move {
            scheme.try_block_primitive(uc, block).map(|p| p.to_shared())
        };
        pc_map.insert(block_index, runtime.spawn(primitive).map(|f| f.unwrap()).shared());
    }
    let pc_map = Arc::new(pc_map);

    for block in block_data {
        let solver      = solver.clone();
        let mesh        = mesh.clone();
        let pc_map      = pc_map.clone();
        let block       = block.clone();
        let block_index = block.index;

        let fv = async move {
            let pn = try_join_3by3(mesh.neighbor_block_indexes(block_index).map_3by3(|i| &pc_map[i])).await?;
            let pe = ndarray_ops::extend_from_neighbor_arrays_2d(&pn, 2, 2, 2, 2);

            if solver.using_tracers() {
                let (fx, fy, vx, vy) = scheme.compute_block_fluxes(&pe, &block, &solver, &mesh, time);
                let vstar_x = vx.unwrap().to_shared();
                let vstar_y = vy.unwrap().to_shared();
                Ok::<_, HydroError>((fx.to_shared(), fy.to_shared(), Some(vstar_x), Some(vstar_y)))
            } else {
                let (fx, fy, _, _) = scheme.compute_block_fluxes(&pe, &block, &solver, &mesh, time);
                Ok::<_, HydroError>((fx.to_shared(), fy.to_shared(), None, None))
            }
        };
        fv_map.insert(block_index, runtime.spawn(fv).map(|fv| fv.unwrap()).shared());
    } 
    let fv_map = Arc::new(fv_map);

    for (solution, block) in state.solution.iter().zip(block_data) {
        let solver   = solver.clone();
        let mesh     = mesh.clone();
        let fv_map   = fv_map.clone();
        let block    = block.clone();
        let solution = solution.clone();

        let s1 = async move {
            let (fx, fy, vx, vy) = if ! solver.need_flux_communication() {
                fv_map[&block.index].clone().await?
            } else {
                let fv_n = try_join_3by3(mesh.neighbor_block_indexes(block.index).map_3by3(|i| &fv_map[i])).await?;
                let fx_n = fv_n.map_3by3(|fv| fv.0.clone());
                let fy_n = fv_n.map_3by3(|fv| fv.1.clone());
                let vx_n = fv_n.map_3by3(|fv| fv.2.clone().unwrap());
                let vy_n = fv_n.map_3by3(|fv| fv.3.clone().unwrap());
                let fx_e = ndarray_ops::extend_from_neighbor_arrays_2d(&fx_n, 1, 1, 1, 1);
                let fy_e = ndarray_ops::extend_from_neighbor_arrays_2d(&fy_n, 1, 1, 1, 1);
                let vx_e = ndarray_ops::extend_from_neighbor_arrays_2d(&vx_n, 1, 1, 1, 1);
                let vy_e = ndarray_ops::extend_from_neighbor_arrays_2d(&vy_n, 1, 1, 1, 1);
                (fx_e.to_shared(), fy_e.to_shared(), Some(vx_e.to_shared()), Some(vy_e.to_shared()))
            };
            Ok::<_, HydroError>(scheme.compute_block_updated_solution(solution, fx, fy, vx, vy, &block, &solver, &mesh, time, dt))
        };
        s1_vec.push(runtime.spawn(s1).map(|f| f.unwrap()))
    }

    let solution = join_all(s1_vec).await
        .iter()
        .cloned()
        .collect::<Result<_, _>>()
        .map_err(|e| e.with_orbital_state(solver.orbital_elements.orbital_state_from_time(time)))?;

    Ok(State {
        time: state.time + dt,
        iteration: state.iteration + 1,
        solution: solution,
    })
}




// ============================================================================
pub fn advance_tokio<H: 'static + Hydrodynamics>(
    mut state:  State<H::Conserved>,
    hydro:      H,
    block_data: &Vec<BlockData<H::Conserved>>,
    mesh:       &Mesh,
    solver:     &Solver,
    dt:         f64,
    fold:       usize,
    runtime:    &tokio::runtime::Runtime) -> Result<State<H::Conserved>, HydroError>
{
    let try_update = |state| try_advance_tokio_rk(state, hydro, block_data, mesh, solver, dt, runtime);
    let rk = solver.runge_kutta();

    for _ in 0..fold {
        state = runtime.block_on(rk.try_advance_async(state, try_update, runtime))?;
        if solver.using_tracers() {
            state = rebin_tracers_sync(state, &mesh, block_data);
        }
    }
    Ok(state)
}




// ============================================================================
fn rebin_tracers_sync<C: Conserved>(state: State<C>, mesh: &Mesh, block_data: &Vec<BlockData<C>>) -> State<C>
{
    let tracer_map: HashMap<_, _> = state.solution.iter().zip(block_data).map(|(s, block)|
    {
        let (on, off) = tracers_on_and_off_block(&s.tracers, &mesh, block.index);
        (block.index, (on, Arc::new(off)))
    }).collect();

    let solution = state.solution.iter().zip(block_data).map(|(s, block)|
    {
        let my_tracers            = tracer_map[&block.index].clone().0;
        let tracers_on_off_n      = mesh.neighbor_block_indexes(block.index).map_3by3(|i| &tracer_map[i]);
        let tracers_to_be_claimed = tracers_on_off_n.map_3by3(|(_, off)| off.clone());
        s.with_tracers(push_new_tracers(my_tracers, tracers_to_be_claimed, &mesh, block.index))
    });

    State {
        time: state.time,
        iteration: state.iteration,
        solution: solution.collect(),
    }
}




// ============================================================================
impl<H: Hydrodynamics> UpdateScheme<H>
{
    fn new(hydro: H) -> Self {
        Self{hydro}
    }

    fn try_block_primitive(
        &self,
        conserved: ArcArray<H::Conserved, Ix2>,
        block: BlockData<H::Conserved>) -> Result<Array<H::Primitive, Ix2>, HydroError>
    {
        let x: Result<Vec<_>, _> = conserved
            .iter()
            .zip(block.cell_centers.iter())
            .map(|(&u, &xy)| self
                .hydro
                .try_to_primitive(u)
                .map_err(|e| e.at_position(xy)))
            .collect();
        Ok(Array::from_shape_vec(conserved.dim(), x?).unwrap())
    }

    fn block_riemann_solutions(&self,
        cell_data: &Array<CellData<H::Primitive>, Ix2>,
        faces:     &ArcArray<(f64, f64), Ix2>,
        dx:        f64,
        dy:        f64,
        solver:    &Solver,
        dir:       Direction,
        time:      f64) -> (Array<H::Conserved, Ix2>, Option<Array<f64, Ix2>>)
    {
        use ndarray::{s, azip};
        use crate::traits::{Primitive};
        use crate::physics::Direction::{X, Y};

        let two_body_state = solver.orbital_elements.orbital_state_from_time(time);

        let face_data = match dir {
            X => azip![
                    cell_data.slice(s![..-1, 1..-1]),
                    cell_data.slice(s![1..,  1..-1]),
                    faces],
            Y => azip![
                    cell_data.slice(s![1..-1, ..-1]),
                    cell_data.slice(s![1..-1, 1.. ]),
                    faces],
        };

        if solver.need_face_velocities() {
            let riemann_solver = |l, r, f|
            {
                let (flux, u) = self.hydro.intercell_flux_plus_state(&solver, l, r, f, dx, dy, &two_body_state, dir);
                let vstar = match dir {
                    X => self.hydro.to_primitive(u).velocity_x(),
                    Y => self.hydro.to_primitive(u).velocity_y(),
                };
                (flux, vstar)
            };
            let fv = face_data.apply_collect(riemann_solver);
            (fv.map(|&(f, _)| f), Some(fv.map(|&(_, v)| v)))

        } else {
            let riemann_solver = |l, r, f| self.hydro.intercell_flux(&solver, l, r, f, dx, dy, &two_body_state, dir);
            (face_data.apply_collect(riemann_solver), None)
        }
    }

    fn compute_block_fluxes(
        &self,
        pe:     &Array<H::Primitive, Ix2>,
        block:  &BlockData<H::Conserved>,
        solver: &Solver,
        mesh:     &Mesh,
        time:   f64) -> (Array<H::Conserved, Ix2>, Array<H::Conserved, Ix2>, Option<Array<f64, Ix2>>, Option<Array<f64, Ix2>>)
    {
        use ndarray::{s, azip};
        use ndarray_ops::map_stencil3;

        // ========================================================================
        let gx = map_stencil3(&pe, Axis(0), |a, b, c| self.hydro.plm_gradient(solver.plm, a, b, c));
        let gy = map_stencil3(&pe, Axis(1), |a, b, c| self.hydro.plm_gradient(solver.plm, a, b, c));
        let dx = mesh.cell_spacing_x();
        let dy = mesh.cell_spacing_y();
        let xf = &block.face_centers_x;
        let yf = &block.face_centers_y;

        // ============================================================================
        let cell_data = azip![
            pe.slice(s![1..-1,1..-1]),
            gx.slice(s![ ..  ,1..-1]),
            gy.slice(s![1..-1, ..  ])]
        .apply_collect(CellData::new);

        let (fx, vx) = self.block_riemann_solutions(&cell_data, xf, dx, dy, solver, Direction::X, time);
        let (fy, vy) = self.block_riemann_solutions(&cell_data, yf, dx, dy, solver, Direction::Y, time);

        (fx, fy, vx, vy)
    }

    fn compute_block_tracer_update(
        &self,
        tracers:  &Vec<Tracer>,
        vstar_x:  ArcArray<f64, Ix2>,
        vstar_y:  ArcArray<f64, Ix2>,
        index:    BlockIndex,
        mesh:     &Mesh,
        dt:       f64) -> Vec<Tracer>
    {
        tracers.into_iter()
               .map(|t| update_tracer(t, &mesh, index, &vstar_x, &vstar_y, 1, dt))
               .collect()
    }

    fn compute_block_updated_solution(
        &self,
        solution: BlockSolution<H::Conserved>,
        fx:       ArcArray<H::Conserved, Ix2>,
        fy:       ArcArray<H::Conserved, Ix2>,
        vx:       Option<ArcArray<f64, Ix2>>,
        vy:       Option<ArcArray<f64, Ix2>>,
        block:    &BlockData<H::Conserved>,
        solver:   &Solver,
        mesh:     &Mesh,
        time:     f64,
        dt:       f64) -> BlockSolution<H::Conserved>
    {
        let dx = mesh.cell_spacing_x();
        let dy = mesh.cell_spacing_y();
        let two_body_state = solver.orbital_elements.orbital_state_from_time(time);

        let mut ds = ItemizedChange::zeros();

        let u1 = ArcArray::from_shape_fn(solution.conserved.dim(), |i| {
            let m = if solver.need_flux_communication() {
                (i.0 + 1, i.1 + 1)
            } else {
                i
            };
            let du = ((fx[(m.0 + 1, m.1)] - fx[m]) / dx +
                      (fy[(m.0, m.1 + 1)] - fy[m]) / dy) * -dt;
            let uc = solution.conserved[i];
            let u0 = block.initial_conserved[i];
            let (x, y)  = block.cell_centers[i];
            let sources = self.hydro.source_terms(&solver, uc, u0, x, y, dt, &two_body_state);

            ds.add_mut(&sources);
            uc + du + sources.total()
        });

        let ds = ds.mul(dx * dy);
        let de = ds.perturbation(time, solver.orbital_elements);

        let new_tracers = if solver.using_tracers()  {
            Arc::new(self.compute_block_tracer_update(&solution.tracers, vx.unwrap(), vy.unwrap(), block.index, &mesh, dt))
        }
        else {
            solution.tracers
        };

        BlockSolution{
            conserved: u1,
            integrated_source_terms: solution.integrated_source_terms.add(&ds),
            orbital_elements_change: solution.orbital_elements_change.add(&de),
            tracers: new_tracers,
        }
    }
}

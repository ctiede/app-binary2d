use libm::{erf, exp, sqrt};
use std::f64::consts::PI;




static G: f64     = 1.0;    // gravitational constant
static M: f64     = 1.0;    // system mass


// ============================================================================
pub trait DiskModel {
    fn validation_message(&self) -> Option<String>;
    fn phi_velocity_squared(&self, r: f64) -> f64;
    fn vertically_integrated_pressure(&self, r: f64) -> f64;
    fn surface_density(&self, r: f64) -> f64;
}




// ============================================================================
pub struct Torus {
    pub mach_number: f64,
    pub softening_length: f64,
    pub mass: f64,
    pub radius: f64,
    pub width: f64,
    pub domain_radius: f64,
    pub gamma: f64,
}

impl DiskModel for Torus {
    fn validation_message(&self) -> Option<String> {
        if self.failure_radius() < self.domain_radius * f64::sqrt(2.0) {
            Some(concat!{
                "equilibrium disk model fails inside the domain, ",
                "use a larger mach_number, larger disk_width, or a smaller domain_radius."
            }.into())
        } else {
            None
        }
    }

    fn phi_velocity_squared(&self, r: f64) -> f64 {
        self.sound_speed_squared(r) * self.real_mach_number_squared(r)
    }

    fn vertically_integrated_pressure(&self, r: f64) -> f64 {
        self.surface_density(r) * self.sound_speed_squared(r) / self.gamma
    }

    fn surface_density(&self, r: f64) -> f64 {
        let r0 = self.radius;
        let dr = self.width;
        self.sigma0() * exp(-((r - r0) / dr).powi(2))
    }
}

impl Torus {

    pub fn failure_radius(&self) -> f64 {
        let ma = self.mach_number;
        let r0 = self.radius;
        let dr = self.width;
        0.5 * (r0 + sqrt(r0 * r0 + 2.0 * dr * dr * (ma * ma * self.gamma - 1.0)))
    }

    fn real_mach_number_squared(&self, r: f64) -> f64 {
        let ma = self.mach_number;
        let rs = self.softening_length;
        ma * ma - ((r * r - 2.0 * rs * rs) / (r * r + rs * rs) - self.dlogrho_dlogr(r)) / self.gamma
    }

    fn kepler_speed_squared(&self, r: f64) -> f64 {
        let rs = self.softening_length;
        G * M * r * r / (r * r + rs * rs).powf(1.5)
    }

    fn sound_speed_squared(&self, r: f64) -> f64 {
        self.kepler_speed_squared(r) / self.mach_number.powi(2)
    }

    fn dlogrho_dlogr(&self, r: f64) -> f64 {
        let r0 = self.radius;
        let dr = self.width;
        -2.0 * r * (r - r0) / dr.powi(2)
    }

    fn sigma0(&self) -> f64 {
        let r0 = self.radius;
        let dr = self.width;
        let md = self.mass;
        let total = PI * dr * dr * (exp(-(r0 / dr).powi(2)) + sqrt(PI) * r0 / dr * (1.0 + erf(r0 / dr)));
        return md / total;
    }
}




// ============================================================================
pub enum InfiniteDisk {
    Flat (FlatDisk),
    Alpha(AlphaDisk),
}

impl DiskModel for InfiniteDisk {

    fn validation_message(&self) -> Option<String> {
        None
    }

    fn phi_velocity_squared(&self, r: f64) -> f64 {
        match self {
            InfiniteDisk::Flat (disk) => disk.phi_velocity_squared(r),
            InfiniteDisk::Alpha(disk) => disk.phi_velocity_squared(r),
        }
    }

    fn vertically_integrated_pressure(&self, r: f64) -> f64 {
        match self {
            InfiniteDisk::Flat (disk) => disk.vertically_integrated_pressure(r),
            InfiniteDisk::Alpha(disk) => disk.vertically_integrated_pressure(r),
        }
    }

    fn surface_density(&self, r: f64) -> f64 {
        match self {
            InfiniteDisk::Flat (disk) => disk.surface_density(r),
            InfiniteDisk::Alpha(disk) => disk.surface_density(r),
        }
    }
}



// ============================================================================
pub struct FlatDisk {
    pub nu: f64,
    pub radius: f64,
    pub mach_number: f64,
    pub softening_length: f64,
    pub gamma: f64,
    pub mdot0: f64,
    pub initial_floor: f64
}

impl DiskModel for FlatDisk {

    fn validation_message(&self) -> Option<String> {
        None
    }

    fn phi_velocity_squared(&self, r: f64) -> f64 {
        self.kepler_speed_squared(r) 
    }

    fn vertically_integrated_pressure(&self, r: f64) -> f64 {
        self.surface_density(r) * self.sound_speed_squared(r) / self.gamma
    }

    fn surface_density(&self, _: f64) -> f64 {
        self.mdot0 / (3. * PI * self.nu)
    }
}

impl FlatDisk {

    fn kepler_speed_squared(&self, r: f64) -> f64 {
        let rs = self.softening_length;
        G * M * r * r / (r * r + rs * rs).powf(1.5)
    }

    fn sound_speed_squared(&self, r: f64) -> f64 {
        self.kepler_speed_squared(r) / self.mach_number.powi(2)
    }
}




// ============================================================================
// 
// Following setup up from Munoz, Miranda, Lai 2019 (only for equal-mass, circular case)
// 
pub struct AlphaDisk {
    pub alpha: f64,
    pub radius: f64,
    pub mach_number: f64,
    pub softening_length: f64,
    pub ell0: f64,
    pub mdot0: f64,
    pub gamma: f64,
    pub initial_floor: f64
}

impl DiskModel for AlphaDisk {

    fn validation_message(&self) -> Option<String> {
        None
    }

    fn phi_velocity_squared(&self, r: f64) -> f64 {
            self.vphi_squared(r) * self.fcavity(r).powi(2)
    }

    fn vertically_integrated_pressure(&self, r: f64) -> f64 {
        self.surface_density(r) * self.sound_speed_squared(r) / self.gamma
    }

    fn surface_density(&self, r: f64) -> f64 {
        self.sigma(r) * self.fcavity(r) + self.initial_floor
    }
}

impl AlphaDisk {

    fn kepler_speed_squared(&self, r: f64) -> f64 {
        let rs = self.softening_length;
        G * M * r * r / (r * r + rs * rs).powf(1.5)
    }

    fn sound_speed_squared(&self, r: f64) -> f64 {
        self.kepler_speed_squared(r) / self.mach_number.powi(2)
    }

    fn sigma(&self, r: f64) -> f64 {
        let rinv_sqrt = (1. / r).sqrt();
        self.sigma0() * rinv_sqrt * (1. - self.ell0 * rinv_sqrt)
    }

    fn dsigma_dr(&self, r: f64) -> f64 {
        self.sigma0() * (-0.5 / r.powf(1.5) + self.ell0 / r.powi(2))
    }

    fn vphi_squared(&self, r: f64) -> f64 {
        let r_inv  = 1. / r;
        let q_term = 1. + 3. / 16. * r_inv * r_inv;
        let p_term = self.sound_speed_squared(r) * (1. - r / self.sigma(r) * self.dsigma_dr(r));
        G * M * r_inv * q_term - p_term
    }

    fn fcavity(&self, r: f64) -> f64 {
        let r0 = self.radius;
        exp(-(r0 / r).powi(4))
    }

    fn sigma0(&self) -> f64 {
        self.mdot0 / (3. * PI * self.alpha / self.mach_number.powi(2))
    }
}




// ============================================================================
pub struct Pringle81 {
}

impl DiskModel for Pringle81 {
    fn validation_message(&self) -> Option<String> {
        None
    }

    fn phi_velocity_squared(&self, _r: f64) -> f64 {
        todo!()
    }

    fn vertically_integrated_pressure(&self, _r: f64) -> f64 {
        todo!()
    }

    fn surface_density(&self, _r: f64) -> f64 {
        todo!()
    }
}

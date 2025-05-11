use std::fmt::Write;

use itertools::Itertools;
use wgpu::ComputePipeline;

use crate::machine_learning::nn::tensor_error::{Result, TensorError};

#[derive(Debug)]
pub struct Kernel {
    // bind_group_layout: BindGroupLayout,
    pub workgroup: Workgroup,
    pub pipeline: ComputePipeline,
}

#[derive(Debug, Clone, Copy)]
pub struct Workgroup {
    pub local_x: u32,
    pub local_y: u32,
    pub local_z: u32,
    pub global_x: u32,
    pub global_y: u32,
    pub global_z: u32,
}

impl From<[u32; 6]> for Workgroup {
    fn from(workgroup_size: [u32; 6]) -> Self {
        Self {
            local_x: workgroup_size[0],
            local_y: workgroup_size[1],
            local_z: workgroup_size[2],
            global_x: workgroup_size[3],
            global_y: workgroup_size[4],
            global_z: workgroup_size[5],
        }
    }
}

pub enum KernelParameter {
    WorkgroupSizeX(u32),
    WorkgroupSizeY(u32),
    StartOffsetX(u32),
    StartOffsetY(u32),
    StartOffsetZ(u32),
    StartOffsetN(u32),
    H(u32),
    W(u32),
    M(u32),
    N(u32),
    K(u32),
    P(u32),
    Q(u32),
    C(u32),
    R(u32),
    S(u32),
    PXL(u32),
    QXL(u32),
    CXL(u32),
    NXL(u32),
    PoolSizeX(u32),
    PoolSizeY(u32),
    UpsampleSizeX(u32),
    UpsampleSizeY(u32),
    Length(u32),
    Alpha(f32),
    GlobalH(u32),
    GlobalW(u32),
    AN(u32),
    AC(u32),
    AH(u32),
    AW(u32),
    BN(u32),
    BC(u32),
    BH(u32),
    BW(u32),
    BroadAN(u32),
    BroadAC(u32),
    BroadAH(u32),
    BroadAW(u32),
    BroadBN(u32),
    BroadBC(u32),
    BroadBH(u32),
    BroadBW(u32),
    Dim0(u32),
    Dim1(u32),
    Dim2(u32),
    Dim3(u32),
    Step(u32),
    Count(u32),
}

pub struct KernelParameters<'a>(pub(super) &'a [KernelParameter]);

impl std::fmt::Display for KernelParameters<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let params = self.0.iter().fold(String::new(), |mut output, p| {
            let _ = write!(output, "{p}");
            output
        });
        write!(f, "{params}")
    }
}

impl std::fmt::Display for KernelParameter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let param = match self {
            KernelParameter::WorkgroupSizeX(value) => {
                format!("const WORKGROUP_SIZE_X = {value}u;\n")
            }
            KernelParameter::WorkgroupSizeY(value) => {
                format!("const WORKGROUP_SIZE_Y = {value}u;\n")
            }
            KernelParameter::StartOffsetX(value) => {
                format!("const START_OFFSET_X = {value}u;\n")
            }
            KernelParameter::StartOffsetY(value) => {
                format!("const START_OFFSET_Y = {value}u;\n")
            }
            KernelParameter::StartOffsetZ(value) => {
                format!("const START_OFFSET_Z = {value}u;\n")
            }
            KernelParameter::StartOffsetN(value) => {
                format!("const START_OFFSET_N = {value}u;\n")
            }
            KernelParameter::M(value) => format!("const M = {value}u;\n"),
            KernelParameter::N(value) => format!("const N = {value}u;\n"),
            KernelParameter::K(value) => format!("const K = {value}u;\n"),
            KernelParameter::P(value) => format!("const P = {value}u;\n"),
            KernelParameter::Q(value) => format!("const Q = {value}u;\n"),
            KernelParameter::C(value) => format!("const C = {value}u;\n"),
            KernelParameter::R(value) => format!("const R = {value}u;\n"),
            KernelParameter::S(value) => format!("const S = {value}u;\n"),
            KernelParameter::PXL(value) => format!("const P_XL = {value}u;\n"),
            KernelParameter::QXL(value) => format!("const Q_XL = {value}u;\n"),
            KernelParameter::CXL(value) => format!("const C_XL = {value}u;\n"),
            KernelParameter::NXL(value) => format!("const N_XL = {value}u;\n"),
            KernelParameter::PoolSizeX(value) => format!("const POOL_SIZE_X = {value}u;\n"),
            KernelParameter::PoolSizeY(value) => format!("const POOL_SIZE_Y = {value}u;\n"),
            KernelParameter::UpsampleSizeX(value) => format!("const UPSAMPLE_SIZE_X = {value}u;\n"),
            KernelParameter::UpsampleSizeY(value) => format!("const UPSAMPLE_SIZE_Y = {value}u;\n"),
            KernelParameter::Length(value) => format!("const LENGTH = {value}u;\n"),
            KernelParameter::Alpha(value) => format!("const ALPHA = {value}f;\n"),
            KernelParameter::GlobalH(value) => format!("const GLOBAL_H = {value}u;\n"),
            KernelParameter::GlobalW(value) => format!("const GLOBAL_W = {value}u;\n"),
            KernelParameter::H(value) => format!("const H = {value}u;\n"),
            KernelParameter::W(value) => format!("const W = {value}u;\n"),
            KernelParameter::AN(value) => format!("const A_N = {value}u;\n"),
            KernelParameter::AC(value) => format!("const A_C = {value}u;\n"),
            KernelParameter::AH(value) => format!("const A_H = {value}u;\n"),
            KernelParameter::AW(value) => format!("const A_W = {value}u;\n"),
            KernelParameter::BN(value) => format!("const B_N = {value}u;\n"),
            KernelParameter::BC(value) => format!("const B_C = {value}u;\n"),
            KernelParameter::BH(value) => format!("const B_H = {value}u;\n"),
            KernelParameter::BW(value) => format!("const B_W = {value}u;\n"),
            KernelParameter::BroadAN(value) => format!("const BROAD_A_N = {value}u;\n"),
            KernelParameter::BroadAC(value) => format!("const BROAD_A_C = {value}u;\n"),
            KernelParameter::BroadAH(value) => format!("const BROAD_A_H = {value}u;\n"),
            KernelParameter::BroadAW(value) => format!("const BROAD_A_W = {value}u;\n"),
            KernelParameter::BroadBN(value) => format!("const BROAD_B_N = {value}u;\n"),
            KernelParameter::BroadBC(value) => format!("const BROAD_B_C = {value}u;\n"),
            KernelParameter::BroadBH(value) => format!("const BROAD_B_H = {value}u;\n"),
            KernelParameter::BroadBW(value) => format!("const BROAD_B_W = {value}u;\n"),
            KernelParameter::Dim0(value) => format!("const DIM0 = {value}u;\n"),
            KernelParameter::Dim1(value) => format!("const DIM1 = {value}u;\n"),
            KernelParameter::Dim2(value) => format!("const DIM2 = {value}u;\n"),
            KernelParameter::Dim3(value) => format!("const DIM3 = {value}u;\n"),
            KernelParameter::Step(value) => format!("const STEP = {value}u;\n"),
            KernelParameter::Count(value) => format!("const COUNT = {value}u;\n"),
        };
        write!(f, "{param}")
    }
}

#[derive(Debug)]
pub enum KernelError {
    InvalidWorkGroup(String),
    InvalidKernel(String),
}

impl Kernel {
    /// Quick hack to modify the workgroup sizes in the kernel source.
    pub fn modify_workgroup_size(source: &str, workgroup: Workgroup) -> Result<String> {
        let mut lines = source.lines();
        if let Some(first_line) = lines.next() {
            if first_line.trim().starts_with("const WORKGROUP_SIZE_X") {
                let first_line = format!("const WORKGROUP_SIZE_X = {}u;\n", workgroup.local_x);
                if let Some(second_line) = lines.next() {
                    if second_line.trim().starts_with("const WORKGROUP_SIZE_Y") {
                        let second_line =
                            format!("const WORKGROUP_SIZE_Y = {}u;\n", workgroup.local_y);
                        let mut modified_source = String::new();
                        modified_source.push_str(&first_line);
                        modified_source.push_str(&second_line);
                        for line in lines {
                            modified_source.push_str(line);
                            modified_source.push('\n');
                        }
                        Ok(modified_source)
                    } else {
                        Err(TensorError::BackendError(
                            "No WORKGROUP_SIZE_Y found in the first line".to_string(),
                        ))
                    }
                } else {
                    Err(TensorError::BackendError("Empty kernel source".to_string()))
                }
            } else {
                Err(TensorError::BackendError(
                    "No WORKGROUP_SIZE_X found in the first line".to_string(),
                ))
            }
        } else {
            Err(TensorError::BackendError("Empty kernel source".to_string()))
        }
    }

    pub fn set_parameters(source: &str, parameters: &KernelParameters) -> String {
        let source_without_dummy_params: String =
            source.lines().skip(parameters.0.len()).join("\n");
        let params: String = parameters
            .0
            .iter()
            .fold(String::new(), |mut params_string, p| {
                let _ = write!(params_string, "{p}");
                params_string
            });
        format!("{}{}", params, source_without_dummy_params)
    }
}

#[cfg(test)]
mod tests {
    use crate::machine_learning::nn::backends::wgpu_backend::kernel::KernelParameter;

    use super::*;
    use super::{Kernel, Workgroup};

    #[test]
    fn modify_kernel_no_workgroups() {
        let source_with_16 = "something;
const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
something"
            .to_string();

        let workgroup = Workgroup {
            local_x: 32,
            local_y: 32,
            local_z: 1,
            global_x: 1,
            global_y: 1,
            global_z: 1,
        };
        let modified_source = Kernel::modify_workgroup_size(&source_with_16, workgroup);
        assert!(modified_source.is_err());
    }

    #[test]
    fn modify_kernel_empty_kernel() {
        let source_with_16 = "".to_string();

        let workgroup = Workgroup {
            local_x: 32,
            local_y: 32,
            local_z: 1,
            global_x: 1,
            global_y: 1,
            global_z: 1,
        };
        let modified_source = Kernel::modify_workgroup_size(&source_with_16, workgroup);
        assert!(modified_source.is_err());
    }

    #[test]
    fn set_kernel_parameters() {
        let kernel_source = include_str!("./add_zero_padding2.wgsl");

        let parameter_list = [
            KernelParameter::WorkgroupSizeX(16),
            KernelParameter::WorkgroupSizeY(16),
            KernelParameter::P(45),
            KernelParameter::Q(46),
            KernelParameter::PXL(66),
            KernelParameter::QXL(565),
        ];

        let parameters = KernelParameters(&parameter_list);
        let modified_source = Kernel::set_parameters(kernel_source, &parameters);
        println!("{}", modified_source);
    }
}

use super::game_model::MazeAction;

pub mod awesom_model;
pub mod burn_model;

pub trait InferenceModel {
    fn predict_move(&self, state: Vec<f32>) -> MazeAction;
}

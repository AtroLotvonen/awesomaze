extern crate awesomlearn;

use awesomlearn::machine_learning::{
    nn::dimensions::Dimensions3D,
    reinforcement_learning::environment::{RLAction, RLEnvironment, RLState, Reward},
};
use rand::{seq::IteratorRandom, Rng};

use super::input::GameInput;
mod recursive_backtracker;

#[derive(Debug, Clone)]
pub struct GameModel {
    level: u32,
    maze: Vec<MazeSquare>,
    bounds: (usize, usize),
    player: PlayerModel,
    cake: CakeModel,
    right_path: Vec<usize>,
    visited_path: Vec<(usize, usize)>,
    cumulative_reward: f32,
}

impl AsRef<GameModel> for GameModel {
    fn as_ref(&self) -> &GameModel {
        self
    }
}

#[derive(Debug, Clone)]
struct PlayerModel {
    location: (usize, usize),
}

#[derive(Debug, Clone)]
struct CakeModel {
    location: (usize, usize),
}

#[derive(Debug, Clone)]
pub enum MazeSquare {
    Wall,
    Floor,
}

#[derive(Clone, Debug, Copy)]
pub enum MazeAction {
    Left,
    Up,
    Right,
    Down,
}

impl RLAction for MazeAction {
    const SIZE: u32 = 4;
}

impl From<f32> for MazeAction {
    fn from(value: f32) -> Self {
        if value <= 0.1 {
            MazeAction::Left
        } else if value <= 1.1 {
            MazeAction::Up
        } else if value <= 2.1 {
            MazeAction::Right
        } else {
            MazeAction::Down
        }
    }
}

impl From<MazeAction> for f32 {
    fn from(val: MazeAction) -> Self {
        match val {
            MazeAction::Left => 0.0,
            MazeAction::Up => 1.0,
            MazeAction::Right => 2.0,
            MazeAction::Down => 3.0,
        }
    }
}

impl From<u32> for MazeAction {
    fn from(value: u32) -> Self {
        if value == 0 {
            MazeAction::Left
        } else if value == 1 {
            MazeAction::Up
        } else if value == 2 {
            MazeAction::Right
        } else {
            MazeAction::Down
        }
    }
}

impl From<MazeAction> for u32 {
    fn from(val: MazeAction) -> Self {
        match val {
            MazeAction::Left => 0,
            MazeAction::Up => 1,
            MazeAction::Right => 2,
            MazeAction::Down => 3,
        }
    }
}

#[derive(Clone, Debug)]
pub struct MazeState<const W: usize, const H: usize>(Vec<f32>);

impl<const W: usize, const H: usize> AsRef<[f32]> for MazeState<W, H> {
    fn as_ref(&self) -> &[f32] {
        &self.0
    }
}

impl<const W: usize, const H: usize> RLState for MazeState<W, H> {
    const SIZE: usize = W * H;
}

#[derive(Clone, Debug)]
pub struct RLGameModel<const W: usize, const H: usize>(pub GameModel);

impl<const W: usize, const H: usize> RLGameModel<W, H> {
    pub fn random_level(&mut self) {
        let mut rng = rand::thread_rng();
        let bound_x = (5..W) //Self::FEATURE_BOUNDS.0)
            .step_by(2)
            .choose(&mut rng)
            .unwrap();
        let bound_y = (5..H) //Self::FEATURE_BOUNDS.1)
            .step_by(2)
            .choose(&mut rng)
            .unwrap();
        let new_level = if self.0.has_game_ended() {
            self.0.level() + 1
        } else if self.0.level == 0 {
            0
        } else {
            self.0.level() - 1
        };
        let new_game = GameModel::new((bound_x, bound_y), new_level);

        self.0 = new_game;
    }
}

impl<const W: usize, const H: usize> RLEnvironment for RLGameModel<W, H> {
    type State = MazeState<W, H>;
    type Action = MazeAction;

    fn step(&mut self, action: &Self::Action) -> (Reward, bool) {
        let player_location = self.0.player_location();

        let new_location = match action {
            MazeAction::Left => (player_location.0 - 1, player_location.1),
            MazeAction::Up => (player_location.0, player_location.1 - 1),
            MazeAction::Right => (player_location.0 + 1, player_location.1),
            MazeAction::Down => (player_location.0, player_location.1 + 1),
        };

        let right_path = if self
            .0
            .is_in_right_path(self.0.location_to_index(new_location))
        {
            0.01
        } else {
            0.0
        };
        let reward = if !self.0.can_move_player(new_location) {
            // If player moved to an invalid square
            -0.075
        } else if new_location == self.0.cake_location() {
            // Finds the cake
            1.0
        } else if self.0.visited_path.contains(&new_location) {
            // Already visited location
            -0.025 + right_path
        } else {
            // just walking here
            -0.01 + right_path
        };
        self.0.try_move_player(new_location);
        self.0.cumulative_reward += reward;
        (reward, self.0.has_game_ended())
    }

    fn reset(&mut self) {
        // if successfully ended go to next level else decrease level
        let mut rng = rand::thread_rng();
        let bound_x = (5..W) //Self::FEATURE_BOUNDS.0)
            .step_by(2)
            .choose(&mut rng)
            .unwrap();
        let bound_y = (5..H) //Self::FEATURE_BOUNDS.1)
            .step_by(2)
            .choose(&mut rng)
            .unwrap();
        let new_level = if self.0.has_game_ended() {
            self.0.level() + 1
        } else if self.0.level == 0 {
            0
        } else {
            self.0.level() - 1
        };

        let mut new_game = GameModel::new((bound_x, bound_y), new_level);
        new_game.set_player_distance(new_game.level() as usize);
        // let new_game = GameModel::new(self.0.bounds(), self.level());
        self.0 = new_game;
    }

    fn init() -> Self {
        let model = GameModel::new((5, 5), 0);
        Self(model)
    }

    fn state(&self) -> Self::State {
        let mut features = vec![0.0; Self::State::SIZE];
        let mut rng = rand::thread_rng();
        let padding = (W - self.0.bounds().0, H - self.0.bounds().1);
        let random_offset = (
            rng.gen_range(0..(padding.0 / 2) + 1) * 2,
            rng.gen_range(0..(padding.1 / 2) + 1) * 2,
        );

        let player_pos = self.0.player_location();
        let goal_pos = self.0.cake_location();

        for y in 0..self.0.bounds().1 {
            for x in 0..self.0.bounds().0 {
                // the feature index is like this for later use if convolutions so it's image like
                let x_pos_feature = x + random_offset.0;
                let y_pos_feature = y + random_offset.1;
                let feature_index = y_pos_feature * W + x_pos_feature;

                if (x, y) == player_pos {
                    features[feature_index] = 1.0;
                } else if (x, y) == goal_pos {
                    features[feature_index] = 0.8;
                } else if let Some(sqr) = self.0.maze_square_at_location((x, y)) {
                    match sqr {
                        MazeSquare::Wall => features[feature_index] = 0.0,
                        MazeSquare::Floor => {
                            if self.0.visited_path.contains(&(x, y)) {
                                features[feature_index] = 0.2;
                            } else {
                                features[feature_index] = 0.4;
                            }
                        }
                    }
                }
            }
        }
        // println!("maze");
        // for line in features.chunks(self.0::FEATURE_BOUNDS.0) {
        //     let line_str = line
        //         .iter()
        //         .map(|x| (x * 10.0) as usize)
        //         .map(|x| x.to_string())
        //         .collect::<String>();
        //     println!("{}", line_str);
        // }
        MazeState(features)
    }

    fn dimensions(&self) -> Dimensions3D {
        Dimensions3D::new(1, H, W)
    }

    fn cumulative_reward(&self) -> Reward {
        self.0.cumulative_reward
    }
}

impl GameModel {
    pub fn new(bounds: (usize, usize), level: u32) -> Self {
        let (maze, starting_position, cake_location, mut right_path) =
            recursive_backtracker::generate_maze(bounds);
        assert!(maze.len() == bounds.0 * bounds.1);
        let player = PlayerModel {
            location: starting_position,
        };
        let cake = CakeModel {
            location: cake_location,
        };
        // right_path.sort(); // no sorting so the right path is a stack!
        right_path.dedup();
        Self {
            level,
            maze,
            bounds,
            player,
            cake,
            right_path,
            visited_path: vec![starting_position],
            cumulative_reward: 0.0,
        }
    }

    pub fn set_player_distance(&mut self, dist: usize) {
        let curr_dist = self.right_path.len() - 1; // right_path includes the cake
        if curr_dist <= dist {
        } else {
            let new_player_position = self.right_path[dist + 1];
            self.right_path.truncate(dist + 1);
            self.player.location = self.index_to_location(new_player_position);
        }
    }

    pub fn level(&self) -> u32 {
        self.level
    }

    pub fn has_game_ended(&self) -> bool {
        self.player_location() == self.cake_location()
    }

    pub fn possible_moves(&self) -> Vec<MazeAction> {
        let mut moves = Vec::with_capacity(4);
        if self.can_move_player((self.player_location().0 - 1, self.player_location().1)) {
            moves.push(MazeAction::Left);
        }
        if self.can_move_player((self.player_location().0, self.player_location().1 - 1)) {
            moves.push(MazeAction::Up);
        }
        if self.can_move_player((self.player_location().0 + 1, self.player_location().1)) {
            moves.push(MazeAction::Right);
        }
        if self.can_move_player((self.player_location().0, self.player_location().1 + 1)) {
            moves.push(MazeAction::Down);
        }
        moves
    }

    pub fn not_possible_moves(&self) -> Vec<MazeAction> {
        let mut moves = Vec::with_capacity(4);
        if !self.can_move_player((self.player_location().0 - 1, self.player_location().1)) {
            moves.push(MazeAction::Left);
        }
        if !self.can_move_player((self.player_location().0, self.player_location().1 - 1)) {
            moves.push(MazeAction::Up);
        }
        if !self.can_move_player((self.player_location().0 + 1, self.player_location().1)) {
            moves.push(MazeAction::Right);
        }
        if !self.can_move_player((self.player_location().0, self.player_location().1 + 1)) {
            moves.push(MazeAction::Down);
        }
        moves
    }

    pub fn maze(&self) -> &Vec<MazeSquare> {
        &self.maze
    }

    pub fn bounds(&self) -> (usize, usize) {
        self.bounds
    }

    pub fn index_to_location(&self, index: usize) -> (usize, usize) {
        let x = index % self.bounds.0;
        let y = index / self.bounds.0;
        (x, y)
    }

    pub fn location_to_index(&self, location: (usize, usize)) -> usize {
        location.1 * self.bounds().0 + location.0
    }

    pub fn player_location(&self) -> (usize, usize) {
        self.player.location
    }

    pub fn cake_location(&self) -> (usize, usize) {
        self.cake.location
    }

    pub fn is_in_right_path(&self, index: usize) -> bool {
        self.right_path.contains(&index)
    }

    pub fn maze_square_at_location(&self, location: (usize, usize)) -> Option<&MazeSquare> {
        if location.0 >= self.bounds().0 || location.1 >= self.bounds().1 {
            return None;
        }
        let index = (location.1)
            .saturating_mul(self.bounds.0)
            .saturating_add(location.0);
        self.maze_square_at_index(index)
    }

    pub fn maze_square_at_index(&self, index: usize) -> Option<&MazeSquare> {
        if index >= self.maze.len() {
            None
        } else {
            Some(&self.maze[index])
        }
    }

    pub fn handle_input(&mut self, user_input: GameInput) {
        let player_location = self.player_location();
        let _valid_input = match user_input {
            GameInput::MoveLeft => self.try_move_player((player_location.0 - 1, player_location.1)),
            GameInput::MoveDown => self.try_move_player((player_location.0, player_location.1 + 1)),
            GameInput::MoveUp => self.try_move_player((player_location.0, player_location.1 - 1)),
            GameInput::MoveRight => {
                self.try_move_player((player_location.0 + 1, player_location.1))
            }
            _ => true,
        };
    }

    fn try_move_player(&mut self, new_location: (usize, usize)) -> bool {
        if let Some(tile) = self.maze_square_at_location(new_location) {
            match tile {
                MazeSquare::Wall => false,
                MazeSquare::Floor => {
                    // update right path
                    let loc_index = self.location_to_index(new_location);
                    let _player_index = self.location_to_index(self.player.location);
                    // if new location is in the right path remove the old location from the path
                    // otherwise add the new location
                    if let Some(next_right_loc) =
                        self.right_path.get(self.right_path.len().saturating_sub(2))
                    {
                        if *next_right_loc == loc_index {
                            self.right_path.pop();
                        } else {
                            self.right_path.push(loc_index);
                        }
                    }
                    self.player.location = new_location;
                    self.visited_path.push(new_location);
                    true
                }
            }
        } else {
            false
        }
    }

    fn can_move_player(&self, new_location: (usize, usize)) -> bool {
        if let Some(tile) = self.maze_square_at_location(new_location) {
            match tile {
                MazeSquare::Wall => false,
                MazeSquare::Floor => true,
            }
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_to_location() {
        let model = GameModel::new((30, 20), 1);
        let index = 63;
        assert_eq!(model.index_to_location(index), (3, 2));
    }

    #[test]
    fn test_location_to_index() {
        let model = GameModel::new((50, 26), 1);
        let location = (15, 34);
        assert_eq!(model.location_to_index(location), 1715);
    }

    #[test]
    fn test_location_to_index_and_index_to_loction() {
        let model = GameModel::new((23, 68), 1);
        let location = (15, 34);
        assert_eq!(
            model.index_to_location(model.location_to_index(location)),
            location
        );
    }
}

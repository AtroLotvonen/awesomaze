use std::fmt::Display;

// use super::game_model::MazeSquare;
use rand::prelude::*;

type Maze = (Vec<MazeSquare>, (usize, usize), (usize, usize), Vec<usize>);

pub fn generate_maze(bounds: (usize, usize)) -> Maze {
    let maze_grid = MazePathMaker::new(bounds);
    (
        maze_grid.maze,
        maze_grid.starting_location,
        maze_grid.cake_location,
        maze_grid.right_path,
    )
}

use super::MazeSquare;

struct MazePathMaker {
    bounds: (usize, usize),
    maze_bounds: (usize, usize),
    starting_location: (usize, usize),
    grid: Vec<CarveSquare>,
    maze: Vec<MazeSquare>,
    cake_location: (usize, usize),
    right_path: Vec<usize>,
}

impl MazePathMaker {
    fn new(maze_bounds: (usize, usize)) -> MazePathMaker {
        // The grid can be half of the actual maze since the walls don't require
        // separate layers when generating the paths
        let bounds = (maze_bounds.0 / 2, maze_bounds.1 / 2);
        let size = bounds.0 * bounds.1;
        let grid = vec![CarveSquare::NotVisited; size];
        let mut rng = rand::thread_rng();
        let starting_x: usize = rng.gen_range(0..bounds.0);
        let starting_y: usize = rng.gen_range(0..bounds.1);
        let starting_location = (starting_x * 2 + 1, starting_y * 2 + 1);

        // Generate maze with full of walltiles
        let maze_size = maze_bounds.0 * maze_bounds.1;
        let maze = vec![MazeSquare::Wall; maze_size];

        // let mut rng = rand::thread_rng();
        let cake_location_x = (1..maze_bounds.0 - 1).step_by(2).choose(&mut rng).unwrap();
        let cake_location_y = (1..maze_bounds.1 - 1).step_by(2).choose(&mut rng).unwrap();
        let cake_location = (cake_location_x, cake_location_y);

        let right_path = Vec::new();

        let mut maze_grid = MazePathMaker {
            bounds,
            maze_bounds,
            starting_location,
            grid,
            maze,
            cake_location,
            right_path,
        };
        // Carve paths
        maze_grid.carve_path_from((starting_x, starting_y));
        maze_grid
    }

    fn square_at_location(&self, location: (usize, usize)) -> Option<&CarveSquare> {
        if location.0 >= self.bounds.0 || location.1 >= self.bounds.1 {
            return None;
        }
        let index = (location.1)
            .saturating_mul(self.bounds.0)
            .saturating_add(location.0);
        self.square_at_index(index)
    }

    fn square_at_index(&self, index: usize) -> Option<&CarveSquare> {
        if index >= self.grid.len() {
            None
        } else {
            Some(&self.grid[index])
        }
    }

    fn square_at_location_mut(&mut self, location: (usize, usize)) -> Option<&mut CarveSquare> {
        if location.0 >= self.bounds.0 || location.1 >= self.bounds.1 {
            return None;
        }
        let index = (location.1)
            .saturating_mul(self.bounds.0)
            .saturating_add(location.0);
        self.square_at_index_mut(index)
    }

    fn square_at_index_mut(&mut self, index: usize) -> Option<&mut CarveSquare> {
        if index >= self.grid.len() {
            None
        } else {
            Some(&mut self.grid[index])
        }
    }

    fn carve_path_from(&mut self, location: (usize, usize)) -> bool {
        let mut in_right_path = false;
        // set location visited
        let current_sqr = self.square_at_location_mut(location).unwrap();
        *current_sqr = CarveSquare::Visited;
        let maze_location = (location.0 * 2 + 1, location.1 * 2 + 1);
        let maze_idx = maze_location.1 * self.maze_bounds.0 + maze_location.0;
        self.maze[maze_idx] = MazeSquare::Floor;
        let is_cake_location = maze_location == self.cake_location;
        if is_cake_location {
            self.right_path.push(maze_idx);
            in_right_path = true;
        }

        loop {
            let s_location = (location.0, location.1 + 1);
            let e_location = (location.0 + 1, location.1);
            let n_location = (location.0, location.1.wrapping_sub(1));
            let w_location = (location.0.wrapping_sub(1), location.1);
            let n_square = self.square_at_location(n_location);
            let e_square = self.square_at_location(e_location);
            let s_square = self.square_at_location(s_location);
            let w_square = self.square_at_location(w_location);

            let locations = [n_location, e_location, s_location, w_location];
            let squares = [n_square, e_square, s_square, w_square];

            let not_visited_adjacent_squares: Vec<(&CarveSquare, (usize, usize))> = squares
                .into_iter()
                .zip(locations)
                .filter_map(|(opt, loc)| opt.map(|sqr| (sqr, loc)))
                .filter(|(sqr, _)| **sqr == CarveSquare::NotVisited)
                .collect();
            match not_visited_adjacent_squares.len() {
                0 => {
                    break;
                }
                _ => {
                    let mut rng = rand::thread_rng();
                    // unwrap because the empty array is tested already
                    let new_location = not_visited_adjacent_squares.choose(&mut rng).unwrap().1;
                    let in_cake_path = self.carve_path_from(new_location);
                    let direction = (
                        new_location.0 as i32 - location.0 as i32,
                        new_location.1 as i32 - location.1 as i32,
                    );

                    let path_idx = (maze_location.1 as i32 + direction.1) as usize
                        * self.maze_bounds.0
                        + (maze_location.0 as i32 + direction.0) as usize;
                    self.maze[path_idx] = MazeSquare::Floor;
                    if in_cake_path {
                        self.right_path.push(path_idx);
                        self.right_path.push(maze_idx);
                        in_right_path = true;
                    }
                }
            }
        }
        in_right_path
    }
}

impl Display for MazePathMaker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for x in 0..self.bounds.0 {
            for y in 0..self.bounds.1 {
                let sqr = match self.square_at_location((x, y)) {
                    None => "E",
                    Some(CarveSquare::Visited) => "V",
                    Some(CarveSquare::NotVisited) => "N",
                };
                write!(f, "{}", sqr)?;
            }
            writeln!(f, "\n")?;
        }
        Ok(())
    }
}

#[derive(Clone, PartialEq)]
enum CarveSquare {
    Visited,
    NotVisited,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generation() {
        generate_maze((20, 20));
    }
}

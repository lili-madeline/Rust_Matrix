use std::fmt;
use std::fmt::Write;
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Index, IndexMut, Mul, Sub};

pub trait Square {}

#[derive(Clone, Copy, PartialEq)]
pub struct Matrix<T, const ROW: usize, const COL: usize>
where
    T: Default + Copy + Debug,
{
    pub len: [usize; 2],
    data: [[T; COL]; ROW],
}

impl<T, const ROW: usize, const COL: usize> Default for Matrix<T, ROW, COL>
where
    T: Default + Copy + Debug,
{
    fn default() -> Self {
        Self {
            len: [ROW, COL],
            data: [[Default::default(); COL]; ROW],
        }
    }
}

impl<T, const ROW: usize, const COL: usize> Matrix<T, ROW, COL>
where
    T: Default + Copy + Debug,
{
    pub const fn new(array: [[T; COL]; ROW]) -> Self {
        Self {
            len: [ROW, COL],
            data: array,
        }
    }

    pub fn transpose(&self) -> Matrix<T, COL, ROW> {
        let mut m = Matrix::<T, COL, ROW>::default();
        m.data.iter_mut().enumerate().for_each(|x| {
            x.1.iter_mut()
                .enumerate()
                .for_each(|y| *y.1 = self.data[y.0][x.0])
        });
        m
    }

    pub fn row(&self, index: usize) -> Matrix<T, 1, COL> {
        Matrix::<T, 1, COL> {
            data: [self.data[index - 1]; 1],
            len: [1, COL],
        }
    }
    pub fn col(&self, index: usize) -> Matrix<T, ROW, 1> {
        Matrix::<T, ROW, 1> {
            data: self.data.map(|x| [x[index - 1]]),
            len: [1, COL],
        }
    }
}

impl<T, const SHARED: usize> Square for Matrix<T, SHARED, SHARED> where T: Default + Copy + Debug {}

impl<T, const ROW: usize, const COL: usize> Debug for Matrix<T, ROW, COL>
where
    T: Default + Copy + Debug,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let size = self
            .data
            .iter()
            .flatten()
            .map(|x| format!("{:?}", x).len())
            .max()
            .unwrap();

        let s: String = self
            .data
            .iter()
            .fold(String::new(),|mut out, row| {
                let col = row.iter().fold(String::new(), |mut out, x| {
                    let _ = write!(out, "{: ^1$?}\t", x, size);
                    out
                });
                let _ = writeln!(out, "{col}");
                out
            });
        write!(f, "\n{}", s)
    }
}

impl<T, const ROW: usize, const COL: usize> Index<[usize; 2]> for Matrix<T, ROW, COL>
where
    T: Default + Copy + Debug,
{
    type Output = T;

    fn index(&self, location: [usize; 2]) -> &Self::Output {
        &self.data[location[0] - 1][location[1] - 1]
    }
}

impl<T, const ROW: usize, const COL: usize> IndexMut<[usize; 2]> for Matrix<T, ROW, COL>
where
    T: Default + Copy + Debug,
{
    fn index_mut(&mut self, location: [usize; 2]) -> &mut Self::Output {
        &mut self.data[location[0] - 1][location[1] - 1]
    }
}

impl<T, const ROW: usize, const COL: usize, const SHARED: usize> Mul<Matrix<T, SHARED, COL>>
    for Matrix<T, ROW, SHARED>
where
    T: Default + Copy + Debug + Mul + Sum<<T as Mul>::Output>,
{
    type Output = Matrix<T, ROW, COL>;

    fn mul(self, rhs: Matrix<T, SHARED, COL>) -> Self::Output {
        let mut m = Matrix::<T, ROW, COL>::default();
        m.data.iter_mut().enumerate().for_each(|x| {
            x.1.iter_mut().enumerate().for_each(|y| {
                *y.1 = self.data[x.0]
                    .iter()
                    .enumerate()
                    .map(|i| *i.1 * rhs.data.map(|j| j[y.0])[i.0])
                    .sum();
            })
        });
        m
    }
}

impl<T, const ROW: usize, const COL: usize> Mul<T> for Matrix<T, ROW, COL>
where
    T: Default + Copy + Debug + Mul + Mul<Output = T>,
{
    type Output = Matrix<T, ROW, COL>;

    fn mul(self, rhs: T) -> Self::Output {
        let mut m = Matrix::<T, ROW, COL>::default();
        m.data.iter_mut().enumerate().for_each(|x| {
            x.1.iter_mut()
                .enumerate()
                .for_each(|y| *y.1 = self.data[x.0][y.0] * rhs)
        });
        m
    }
}

impl<T, const ROW: usize, const COL: usize> Add<Matrix<T, ROW, COL>> for Matrix<T, ROW, COL>
where
    T: Default + Copy + Debug + Add<Output = T>,
{
    type Output = Matrix<T, ROW, COL>;

    fn add(self, rhs: Matrix<T, ROW, COL>) -> Self::Output {
        let mut m = Matrix::<T, ROW, COL>::default();
        m.data.iter_mut().enumerate().for_each(|x| {
            x.1.iter_mut()
                .enumerate()
                .for_each(|y| *y.1 = self.data[x.0][y.0] + rhs.data[x.0][y.0])
        });
        m
    }
}

impl<T, const ROW: usize, const COL: usize> Sub<Matrix<T, ROW, COL>> for Matrix<T, ROW, COL>
where
    T: Default + Copy + Debug + Sub<Output = T>,
{
    type Output = Matrix<T, ROW, COL>;

    fn sub(self, rhs: Matrix<T, ROW, COL>) -> Self::Output {
        let mut m = Matrix::<T, ROW, COL>::default();
        m.data.iter_mut().enumerate().for_each(|x| {
            x.1.iter_mut()
                .enumerate()
                .for_each(|y| *y.1 = self.data[x.0][y.0] - rhs.data[x.0][y.0])
        });
        m
    }
}

macro_rules! impl_mul {
    ($($x:ty),*) => {
        $(
        impl<const ROW: usize, const COL: usize> Mul<Matrix<$x, ROW, COL>> for $x
        where
            Matrix<$x, ROW, COL>: Mul<$x, Output = Matrix<$x, ROW, COL>>,
        {
            type Output = Matrix<$x, ROW, COL>;

            fn mul(self, rhs: Matrix<$x, ROW, COL>) -> Self::Output {
                rhs * self
            }
        }
        )*
    }
}

impl_mul![u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64, bool];

#[macro_export]
macro_rules! matrix {
    ($($($x:expr)*);*) => {
        Matrix::new([$([$($x,)*],)*])
    };
    ($($($x:expr)*,);*) => {
        Matrix::new([$([$($x,)*],)*])
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    const MATRIX_SQUARE: Matrix<f64, 2, 2> = matrix![1.0 2.0; 3.0 4.0];
    const MATRIX_NON_SQUARE_1: Matrix<f64, 3, 2> = matrix![1.0 2.0; 3.0 4.0; 5.0 6.0];
    const MATRIX_NON_SQUARE_2: Matrix<f64, 2, 3> = matrix![1.0 2.0 3.0; 4.0 5.0 6.0];
    const MATRIX_NON_SQUARE_3: Matrix<f64, 2, 1> = matrix![1.0; 2.0];

    #[test]
    fn transpose() {
        assert_eq!(MATRIX_NON_SQUARE_2.transpose(), matrix![1.0 4.0; 2.0 5.0; 3.0 6.0]);
    }

    #[test]
    fn debug() {
        println!("{:?}", MATRIX_SQUARE);
    }

    #[test]
    fn add() {
        assert_eq!(MATRIX_SQUARE + MATRIX_SQUARE, matrix![2.0 4.0; 6.0 8.0]);
    }

    #[test]
    fn sub() {
        assert_eq!(MATRIX_SQUARE - MATRIX_SQUARE, matrix![0.0 0.0; 0.0 0.0]);
    }

    //noinspection RsTypeCheck
    #[test]
    fn mul() {
        assert_eq!(2.0 * MATRIX_SQUARE, matrix![2.0 4.0; 6.0 8.0]);
        assert_eq!(MATRIX_SQUARE * MATRIX_SQUARE, matrix![7.0 10.0; 15.0 22.0]);
        assert_eq!(
            MATRIX_NON_SQUARE_1 * MATRIX_NON_SQUARE_2,
            matrix![9.0 12.0 15.0; 19.0 26.0 33.0; 29.0 40.0 51.0]
        );
        assert_eq!(
            MATRIX_NON_SQUARE_1 * MATRIX_NON_SQUARE_3,
            matrix![5.0; 11.0; 17.0]
        );
    }

    #[test]
    fn index() {
        assert_eq!(MATRIX_SQUARE[[1, 2]], 2.0);
    }

    #[test]
    fn index_mut() {
        let mut m = MATRIX_SQUARE;
        m[[1, 2]] = 0.0;
        assert_eq!(m, matrix![1.0 0.0; 3.0 4.0]);
    }
}

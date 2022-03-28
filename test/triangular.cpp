// This file is triangularView of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2008-2009 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifdef EIGEN_TEST_PART_100
#  define EIGEN_NO_DEPRECATED_WARNING
#endif

#include "main.h"


template<typename MatrixType> void triangular_deprecated(const MatrixType &m)
{
  Index rows = m.rows();
  Index cols = m.cols();
  MatrixType m1, m2, m3, m4;
  m1.setRandom(rows,cols);
  m2.setRandom(rows,cols);
  m3 = m1; m4 = m2;
  // deprecated method:
  m1.upperTriangularView().swap(m2);
  // use this method instead:
  m3.upperTriangularView().swap(m4.upperTriangularView());
  VERIFY_IS_APPROX(m1,m3);
  VERIFY_IS_APPROX(m2,m4);
  // deprecated method:
  m1.lowerTriangularView().swap(m4);
  // use this method instead:
  m3.lowerTriangularView().swap(m2.lowerTriangularView());
  VERIFY_IS_APPROX(m1,m3);
  VERIFY_IS_APPROX(m2,m4);
}


template<typename MatrixType> void triangular_square(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  typedef Matrix<Scalar, MatrixType::RowsAtCompileTime, 1> VectorType;

  RealScalar largerEps = 10*test_precision<RealScalar>();

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols),
             m4(rows, cols),
             r1(rows, cols),
             r2(rows, cols);
  VectorType v2 = VectorType::Random(rows);

  MatrixType m1up = m1.upperTriangularView();
  MatrixType m2up = m2.upperTriangularView();

  if (rows*cols>1)
  {
    VERIFY(m1up.isUpperTriangular());
    VERIFY(m2up.transpose().isLowerTriangular());
    VERIFY(!m2.isLowerTriangular());
  }

//   VERIFY_IS_APPROX(m1up.transpose() * m2, m1.upper().transpose().lower() * m2);

  // test overloaded operator+=
  r1.setZero();
  r2.setZero();
  r1.upperTriangularView() +=  m1;
  r2 += m1up;
  VERIFY_IS_APPROX(r1,r2);

  // test overloaded operator=
  m1.setZero();
  m1.upperTriangularView() = m2.transpose() + m2;
  m3 = m2.transpose() + m2;
  VERIFY_IS_APPROX(m3.lowerTriangularView().transpose().toDenseMatrix(), m1);

  // test overloaded operator=
  m1.setZero();
  m1.lowerTriangularView() = m2.transpose() + m2;
  VERIFY_IS_APPROX(m3.lowerTriangularView().toDenseMatrix(), m1);

  VERIFY_IS_APPROX(m3.lowerTriangularView().conjugate().toDenseMatrix(),
                   m3.conjugate().lowerTriangularView().toDenseMatrix());

  m1 = MatrixType::Random(rows, cols);
  for (int i=0; i<rows; ++i)
    while (numext::abs2(m1(i,i))<RealScalar(1e-1)) m1(i,i) = internal::random<Scalar>();

  Transpose<MatrixType> trm4(m4);
  // test back and forward substitution with a vector as the rhs
  m3 = m1.upperTriangularView();
  VERIFY(v2.isApprox(m3.adjoint() * (m1.adjoint().lowerTriangularView().solve(v2)), largerEps));
  m3 = m1.lowerTriangularView();
  VERIFY(v2.isApprox(m3.transpose() * (m1.transpose().upperTriangularView().solve(v2)), largerEps));
  m3 = m1.upperTriangularView();
  VERIFY(v2.isApprox(m3 * (m1.upperTriangularView().solve(v2)), largerEps));
  m3 = m1.lowerTriangularView();
  VERIFY(v2.isApprox(m3.conjugate() * (m1.conjugate().lowerTriangularView().solve(v2)), largerEps));

  // test back and forward substitution with a matrix as the rhs
  m3 = m1.upperTriangularView();
  VERIFY(m2.isApprox(m3.adjoint() * (m1.adjoint().lowerTriangularView().solve(m2)), largerEps));
  m3 = m1.lowerTriangularView();
  VERIFY(m2.isApprox(m3.transpose() * (m1.transpose().upperTriangularView().solve(m2)), largerEps));
  m3 = m1.upperTriangularView();
  VERIFY(m2.isApprox(m3 * (m1.upperTriangularView().solve(m2)), largerEps));
  m3 = m1.lowerTriangularView();
  VERIFY(m2.isApprox(m3.conjugate() * (m1.conjugate().lowerTriangularView().solve(m2)), largerEps));

  // check M * inv(L) using in place API
  m4 = m3;
  m1.transpose().upperTriangularView().solveInPlace(trm4);
  VERIFY_IS_APPROX(m4 * m1.lowerTriangularView(), m3);

  // check M * inv(U) using in place API
  m3 = m1.upperTriangularView();
  m4 = m3;
  m3.transpose().lowerTriangularView().solveInPlace(trm4);
  VERIFY_IS_APPROX(m4 * m1.upperTriangularView(), m3);

  // check solve with unit diagonal
  m3 = m1.unitUpperTriangularView();
  VERIFY(m2.isApprox(m3 * (m1.unitUpperTriangularView().solve(m2)), largerEps));

//   VERIFY((  m1.upperTriangularView()
//           * m2.upperTriangularView()).isUpperTriangular());

  // test swap
  m1.setOnes();
  m2.setZero();
  m2.upperTriangularView().swap(m1.upperTriangularView());
  m3.setZero();
  m3.upperTriangularView().setOnes();
  VERIFY_IS_APPROX(m2,m3);

  m1.setRandom();
  m3 = m1.upperTriangularView();
  Matrix<Scalar, MatrixType::ColsAtCompileTime, Dynamic> m5(cols, internal::random<int>(1,20));  m5.setRandom();
  Matrix<Scalar, Dynamic, MatrixType::RowsAtCompileTime> m6(internal::random<int>(1,20), rows);  m6.setRandom();
  VERIFY_IS_APPROX(m1.upperTriangularView() * m5, m3*m5);
  VERIFY_IS_APPROX(m6*m1.upperTriangularView(), m6*m3);

  m1up = m1.upperTriangularView();
  VERIFY_IS_APPROX(m1.upperSelfadjointView().upperTriangularView().toDenseMatrix(), m1up);
  VERIFY_IS_APPROX(m1up.upperSelfadjointView().upperTriangularView().toDenseMatrix(), m1up);
  VERIFY_IS_APPROX(m1.upperSelfadjointView().lowerTriangularView().toDenseMatrix(), m1up.adjoint());
  VERIFY_IS_APPROX(m1up.upperSelfadjointView().lowerTriangularView().toDenseMatrix(), m1up.adjoint());

  VERIFY_IS_APPROX(m1.upperSelfadjointView().diagonal(), m1.diagonal());

  m3.setRandom();
  const MatrixType& m3c(m3);
  VERIFY( is_same_type(m3c.lowerTriangularView(),m3.lowerTriangularView().template conjugateIf<false>()) );
  VERIFY( is_same_type(m3c.lowerTriangularView().conjugate(),m3.lowerTriangularView().template conjugateIf<true>()) );
  VERIFY_IS_APPROX(m3.lowerTriangularView().template conjugateIf<true>().toDenseMatrix(),
                   m3.conjugate().lowerTriangularView().toDenseMatrix());
  VERIFY_IS_APPROX(m3.lowerTriangularView().template conjugateIf<false>().toDenseMatrix(),
                   m3.lowerTriangularView().toDenseMatrix());

  VERIFY( is_same_type(m3c.lowerSelfadjointView(),m3.lowerSelfadjointView().template conjugateIf<false>()) );
  VERIFY( is_same_type(m3c.lowerSelfadjointView().conjugate(),m3.lowerSelfadjointView().template conjugateIf<true>()) );
  VERIFY_IS_APPROX(m3.lowerSelfadjointView().template conjugateIf<true>().toDenseMatrix(),
                   m3.conjugate().lowerSelfadjointView().toDenseMatrix());
  VERIFY_IS_APPROX(m3.lowerSelfadjointView().template conjugateIf<false>().toDenseMatrix(),
                   m3.lowerSelfadjointView().toDenseMatrix());

}


template<typename MatrixType> void triangular_rect(const MatrixType& m)
{
  typedef typename MatrixType::Scalar Scalar;
  typedef typename NumTraits<Scalar>::Real RealScalar;
  enum { Rows =  MatrixType::RowsAtCompileTime, Cols =  MatrixType::ColsAtCompileTime };

  Index rows = m.rows();
  Index cols = m.cols();

  MatrixType m1 = MatrixType::Random(rows, cols),
             m2 = MatrixType::Random(rows, cols),
             m3(rows, cols),
             m4(rows, cols),
             r1(rows, cols),
             r2(rows, cols);

  MatrixType m1up = m1.upperTriangularView();
  MatrixType m2up = m2.upperTriangularView();

  if (rows>1 && cols>1)
  {
    VERIFY(m1up.isUpperTriangular());
    VERIFY(m2up.transpose().isLowerTriangular());
    VERIFY(!m2.isLowerTriangular());
  }

  // test overloaded operator+=
  r1.setZero();
  r2.setZero();
  r1.upperTriangularView() +=  m1;
  r2 += m1up;
  VERIFY_IS_APPROX(r1,r2);

  // test overloaded operator=
  m1.setZero();
  m1.upperTriangularView() = 3 * m2;
  m3 = 3 * m2;
  VERIFY_IS_APPROX(m3.upperTriangularView().toDenseMatrix(), m1);


  m1.setZero();
  m1.lowerTriangularView() = 3 * m2;
  VERIFY_IS_APPROX(m3.lowerTriangularView().toDenseMatrix(), m1);

  m1.setZero();
  m1.strictlyUpperTriangularView() = 3 * m2;
  VERIFY_IS_APPROX(m3.strictlyUpperTriangularView().toDenseMatrix(), m1);


  m1.setZero();
  m1.strictlyLowerTriangularView() = 3 * m2;
  VERIFY_IS_APPROX(m3.strictlyLowerTriangularView().toDenseMatrix(), m1);
  m1.setRandom();
  m2 = m1.upperTriangularView();
  VERIFY(m2.isUpperTriangular());
  VERIFY(!m2.isLowerTriangular());
  m2 = m1.strictlyUpperTriangularView();
  VERIFY(m2.isUpperTriangular());
  VERIFY(m2.diagonal().isMuchSmallerThan(RealScalar(1)));
  m2 = m1.unitUpperTriangularView();
  VERIFY(m2.isUpperTriangular());
  m2.diagonal().array() -= Scalar(1);
  VERIFY(m2.diagonal().isMuchSmallerThan(RealScalar(1)));
  m2 = m1.lowerTriangularView();
  VERIFY(m2.isLowerTriangular());
  VERIFY(!m2.isUpperTriangular());
  m2 = m1.strictlyLowerTriangularView();
  VERIFY(m2.isLowerTriangular());
  VERIFY(m2.diagonal().isMuchSmallerThan(RealScalar(1)));
  m2 = m1.unitLowerTriangularView();
  VERIFY(m2.isLowerTriangular());
  m2.diagonal().array() -= Scalar(1);
  VERIFY(m2.diagonal().isMuchSmallerThan(RealScalar(1)));
  // test swap
  m1.setOnes();
  m2.setZero();
  m2.upperTriangularView().swap(m1.upperTriangularView());
  m3.setZero();
  m3.upperTriangularView().setOnes();
  VERIFY_IS_APPROX(m2,m3);
}

void bug_159()
{
  Matrix3d m = Matrix3d::Random().triangularView<Lower>();
  EIGEN_UNUSED_VARIABLE(m)
}

EIGEN_DECLARE_TEST(triangular)
{
  int maxsize = (std::min)(EIGEN_TEST_MAX_SIZE,20);
  for(int i = 0; i < g_repeat ; i++)
  {
    int r = internal::random<int>(2,maxsize); TEST_SET_BUT_UNUSED_VARIABLE(r)
    int c = internal::random<int>(2,maxsize); TEST_SET_BUT_UNUSED_VARIABLE(c)

    CALL_SUBTEST_1( triangular_square(Matrix<float, 1, 1>()) );
    CALL_SUBTEST_2( triangular_square(Matrix<float, 2, 2>()) );
    CALL_SUBTEST_3( triangular_square(Matrix3d()) );
    CALL_SUBTEST_4( triangular_square(Matrix<std::complex<float>,8, 8>()) );
    CALL_SUBTEST_5( triangular_square(MatrixXcd(r,r)) );
    CALL_SUBTEST_6( triangular_square(Matrix<float,Dynamic,Dynamic,RowMajor>(r, r)) );

    CALL_SUBTEST_7( triangular_rect(Matrix<float, 4, 5>()) );
    CALL_SUBTEST_8( triangular_rect(Matrix<double, 6, 2>()) );
    CALL_SUBTEST_9( triangular_rect(MatrixXcf(r, c)) );
    CALL_SUBTEST_5( triangular_rect(MatrixXcd(r, c)) );
    CALL_SUBTEST_6( triangular_rect(Matrix<float,Dynamic,Dynamic,RowMajor>(r, c)) );

    CALL_SUBTEST_100( triangular_deprecated(Matrix<float, 5, 7>()) );
    CALL_SUBTEST_100( triangular_deprecated(MatrixXd(r,c)) );
  }
  
  CALL_SUBTEST_1( bug_159() );
}

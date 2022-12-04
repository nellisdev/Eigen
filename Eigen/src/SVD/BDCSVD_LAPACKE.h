// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2022 Melven Roehrig-Zoellner <Melven.Roehrig-Zoellner@DLR.de>
// Copyright (c) 2011, Intel Corporation. All rights reserved.
//
// This file is based on the JacobiSVD_LAPACKE.h originally from Intel -
// see license notice below:
/*
 Redistribution and use in source and binary forms, with or without modification,
 are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
 * Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.
 * Neither the name of Intel Corporation nor the names of its contributors may
   be used to endorse or promote products derived from this software without
   specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
 ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 ********************************************************************************
 *   Content : Eigen bindings to LAPACKe
 *    Singular Value Decomposition - SVD (divide and conquer variant)
 ********************************************************************************
*/
#ifndef EIGEN_BDCSVD_LAPACKE_H
#define EIGEN_BDCSVD_LAPACKE_H

namespace Eigen {

/** \internal Specialization for the data types supported by LAPACKe */

#define EIGEN_LAPACKE_SDD(EIGTYPE, LAPACKE_TYPE, LAPACKE_RTYPE, LAPACKE_PREFIX, EIGCOLROW, LAPACKE_COLROW, OPTIONS) \
template<> inline \
BDCSVD<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>, OPTIONS>& \
BDCSVD<Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>, OPTIONS>::compute_impl(const Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic>& matrix, \
                                                                                                 unsigned int computationOptions) \
{ \
  typedef Matrix<EIGTYPE, Dynamic, Dynamic, EIGCOLROW, Dynamic, Dynamic> MatrixType; \
  /*typedef MatrixType::Scalar Scalar;*/ \
  /*typedef MatrixType::RealScalar RealScalar;*/ \
  allocate(matrix.rows(), matrix.cols(), computationOptions); \
\
  /*const RealScalar precision = RealScalar(2) * NumTraits<Scalar>::epsilon();*/ \
  m_nonzeroSingularValues = m_diagSize; \
\
  lapack_int lda, ldu, ldvt; \
  lapack_int matrix_order = LAPACKE_COLROW; \
  char jobz; \
  LAPACKE_TYPE *u, *vt, dummy; \
  jobz  = (m_computeFullU || m_computeFullV) ? 'A' : (m_computeThinU || m_computeThinV) ? 'S' : 'N'; \
  lapack_int u_cols = (jobz == 'A') ? internal::convert_index<lapack_int>(m_rows) : (jobz == 'S') ? internal::convert_index<lapack_int>(m_diagSize) : 1; \
  lapack_int vt_rows = (jobz == 'A') ? internal::convert_index<lapack_int>(m_cols) : (jobz == 'S') ? internal::convert_index<lapack_int>(m_diagSize) : 1; \
  MatrixType localU; \
  if (computeU() && !(m_computeThinU && m_computeFullV) ) { \
    ldu  = internal::convert_index<lapack_int>(m_matrixU.outerStride()); \
    u    = (LAPACKE_TYPE*)m_matrixU.data(); \
  } else if (computeV()) { \
    localU.resize(m_rows, u_cols);\
    ldu  = internal::convert_index<lapack_int>(localU.outerStride()); \
    u    = (LAPACKE_TYPE*)localU.data(); \
  } else { ldu=1; u=&dummy; }\
  MatrixType localV; \
  if (computeU() || computeV()) { \
    localV.resize(vt_rows, m_cols); \
    ldvt  = internal::convert_index<lapack_int>(localV.outerStride()); \
    vt   = (LAPACKE_TYPE*)localV.data(); \
  } else { ldvt=1; vt=&dummy; }\
  MatrixType m_temp; m_temp = matrix; \
  lda = internal::convert_index<lapack_int>(m_temp.outerStride()); \
  LAPACKE_##LAPACKE_PREFIX##gesdd( matrix_order, jobz, internal::convert_index<lapack_int>(m_rows), internal::convert_index<lapack_int>(m_cols), (LAPACKE_TYPE*)m_temp.data(), lda, (LAPACKE_RTYPE*)m_singularValues.data(), u, ldu, vt, ldvt); \
  if (m_computeThinU && m_computeFullV) { \
    m_matrixU = localU.leftCols(m_matrixU.cols());\
  } \
  if (computeV()) { \
    m_matrixV = localV.adjoint().leftCols(m_matrixV.cols()); \
  } \
  m_isInitialized = true; \
  return *this; \
}

#define EIGEN_LAPACK_SDD_OPTIONS(OPTIONS) \
  EIGEN_LAPACKE_SDD(double,   double,                double, d, ColMajor, LAPACK_COL_MAJOR, OPTIONS) \
  EIGEN_LAPACKE_SDD(float,    float,                 float , s, ColMajor, LAPACK_COL_MAJOR, OPTIONS) \
  EIGEN_LAPACKE_SDD(dcomplex, lapack_complex_double, double, z, ColMajor, LAPACK_COL_MAJOR, OPTIONS) \
  EIGEN_LAPACKE_SDD(scomplex, lapack_complex_float,  float , c, ColMajor, LAPACK_COL_MAJOR, OPTIONS) \
\
  EIGEN_LAPACKE_SDD(double,   double,                double, d, RowMajor, LAPACK_ROW_MAJOR, OPTIONS) \
  EIGEN_LAPACKE_SDD(float,    float,                 float , s, RowMajor, LAPACK_ROW_MAJOR, OPTIONS) \
  EIGEN_LAPACKE_SDD(dcomplex, lapack_complex_double, double, z, RowMajor, LAPACK_ROW_MAJOR, OPTIONS) \
  EIGEN_LAPACKE_SDD(scomplex, lapack_complex_float,  float , c, RowMajor, LAPACK_ROW_MAJOR, OPTIONS)

EIGEN_LAPACK_SDD_OPTIONS(0)
EIGEN_LAPACK_SDD_OPTIONS(ComputeThinU)
EIGEN_LAPACK_SDD_OPTIONS(ComputeThinV)
EIGEN_LAPACK_SDD_OPTIONS(ComputeFullU)
EIGEN_LAPACK_SDD_OPTIONS(ComputeFullV)
EIGEN_LAPACK_SDD_OPTIONS(ComputeThinU | ComputeThinV)
EIGEN_LAPACK_SDD_OPTIONS(ComputeFullU | ComputeFullV)
EIGEN_LAPACK_SDD_OPTIONS(ComputeThinU | ComputeFullV)
EIGEN_LAPACK_SDD_OPTIONS(ComputeFullU | ComputeThinV)

} // end namespace Eigen

#endif // EIGEN_BDCSVD_LAPACKE_H

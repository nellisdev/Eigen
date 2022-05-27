// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2011-2015 Gael Guennebaud <gael.guennebaud@inria.fr>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.


static long int nb_transposed_copies;
#define EIGEN_SPARSE_TRANSPOSED_COPY_PLUGIN {nb_transposed_copies++;}
#define VERIFY_TRANSPOSITION_COUNT(XPR,N) {\
    nb_transposed_copies = 0; \
    XPR; \
    if(nb_transposed_copies!=N) std::cerr << "nb_transposed_copies == " << nb_transposed_copies << "\n"; \
    VERIFY( (#XPR) && nb_transposed_copies==N ); \
  }

#include "sparse.h"

template<typename T>
bool is_sorted(const T& mat) {
  for(Index k = 0; k<mat.outerSize(); ++k)
  {
    Index prev = -1;
    for(typename T::InnerIterator it(mat,k); it; ++it)
    {
      if(prev>=it.index())
        return false;
      prev = it.index();
    }
  }
  return true;
}

template<typename T>
typename internal::nested_eval<T,1>::type eval(const T &xpr)
{
  VERIFY( int(internal::nested_eval<T,1>::type::Flags&RowMajorBit) == int(internal::evaluator<T>::Flags&RowMajorBit) );
  return xpr;
}

template<int OtherStorage, typename SparseMatrixType> void sparse_permutations(const SparseMatrixType& ref)
{
  const Index rows = ref.rows();
  const Index cols = ref.cols();
  typedef typename SparseMatrixType::Scalar Scalar;
  typedef typename SparseMatrixType::StorageIndex StorageIndex;
  typedef SparseMatrix<Scalar, OtherStorage, StorageIndex> OtherSparseMatrixType;
  typedef Matrix<Scalar,Dynamic,Dynamic> DenseMatrix;
  typedef Matrix<StorageIndex,Dynamic,1> VectorI;
//   bool IsRowMajor1 = SparseMatrixType::IsRowMajor;
//   bool IsRowMajor2 = OtherSparseMatrixType::IsRowMajor;
  
  double density = (std::max)(8./static_cast<double>(rows*cols), 0.01);
  
  SparseMatrixType mat(rows, cols), up(rows,cols), lo(rows,cols);
  OtherSparseMatrixType res;
  DenseMatrix mat_d = DenseMatrix::Zero(rows, cols), up_sym_d, lo_sym_d, res_d;
  
  initSparse<Scalar>(density, mat_d, mat, 0);

  up = mat.upperTriangularView();
  lo = mat.lowerTriangularView();
  
  up_sym_d = mat_d.upperSelfadjointView();
  lo_sym_d = mat_d.lowerSelfadjointView();
  
  VERIFY_IS_APPROX(mat, mat_d);
  VERIFY_IS_APPROX(up, DenseMatrix(mat_d.upperTriangularView()));
  VERIFY_IS_APPROX(lo, DenseMatrix(mat_d.lowerTriangularView()));
  
  PermutationMatrix<Dynamic> p, p_null;
  VectorI pi;
  randomPermutationVector(pi, cols);
  p.indices() = pi;

  VERIFY( is_sorted( ::eval(mat*p) ));
  VERIFY( is_sorted( res = mat*p ));
  VERIFY_TRANSPOSITION_COUNT( ::eval(mat*p), 0);
  //VERIFY_TRANSPOSITION_COUNT( res = mat*p, IsRowMajor ? 1 : 0 );
  res_d = mat_d*p;
  VERIFY(res.isApprox(res_d) && "mat*p");

  VERIFY( is_sorted( ::eval(p*mat) ));
  VERIFY( is_sorted( res = p*mat ));
  VERIFY_TRANSPOSITION_COUNT( ::eval(p*mat), 0);
  res_d = p*mat_d;
  VERIFY(res.isApprox(res_d) && "p*mat");

  VERIFY( is_sorted( (mat*p).eval() ));
  VERIFY( is_sorted( res = mat*p.inverse() ));
  VERIFY_TRANSPOSITION_COUNT( ::eval(mat*p.inverse()), 0);
  res_d = mat*p.inverse();
  VERIFY(res.isApprox(res_d) && "mat*inv(p)");

  VERIFY( is_sorted( (p*mat+p*mat).eval() ));
  VERIFY( is_sorted( res = p.inverse()*mat ));
  VERIFY_TRANSPOSITION_COUNT( ::eval(p.inverse()*mat), 0);
  res_d = p.inverse()*mat_d;
  VERIFY(res.isApprox(res_d) && "inv(p)*mat");

  VERIFY( is_sorted( (p * mat * p.inverse()).eval() ));
  VERIFY( is_sorted( res = mat.twistedBy(p) ));
  VERIFY_TRANSPOSITION_COUNT( ::eval(p * mat * p.inverse()), 0);
  res_d = (p * mat_d) * p.inverse();
  VERIFY(res.isApprox(res_d) && "p*mat*inv(p)");

  
  VERIFY( is_sorted( res = mat.upperSelfadjointView().twistedBy(p_null) ));
  res_d = up_sym_d;
  VERIFY(res.isApprox(res_d) && "full selfadjoint upper to full");
  
  VERIFY( is_sorted( res = mat.lowerSelfadjointView().twistedBy(p_null) ));
  res_d = lo_sym_d;
  VERIFY(res.isApprox(res_d) && "full selfadjoint lower to full");
  
  
  VERIFY( is_sorted( res = up.upperSelfadjointView().twistedBy(p_null) ));
  res_d = up_sym_d;
  VERIFY(res.isApprox(res_d) && "upper selfadjoint to full");
  
  VERIFY( is_sorted( res = lo.lowerSelfadjointView().twistedBy(p_null) ));
  res_d = lo_sym_d;
  VERIFY(res.isApprox(res_d) && "lower selfadjoint full");


  VERIFY( is_sorted( res = mat.upperSelfadjointView() ));
  res_d = up_sym_d;
  VERIFY(res.isApprox(res_d) && "full selfadjoint upper to full");

  VERIFY( is_sorted( res = mat.lowerSelfadjointView() ));
  res_d = lo_sym_d;
  VERIFY(res.isApprox(res_d) && "full selfadjoint lower to full");

  VERIFY( is_sorted( res = up.upperSelfadjointView() ));
  res_d = up_sym_d;
  VERIFY(res.isApprox(res_d) && "upper selfadjoint to full");

  VERIFY( is_sorted( res = lo.lowerSelfadjointView() ));
  res_d = lo_sym_d;
  VERIFY(res.isApprox(res_d) && "lower selfadjoint full");


  res.upperSelfadjointView() = mat.upperSelfadjointView();
  res_d = up_sym_d.upperTriangularView();
  VERIFY(res.isApprox(res_d) && "full selfadjoint upper to upper");

  res.lowerSelfadjointView() = mat.upperSelfadjointView();
  res_d = up_sym_d.lowerTriangularView();
  VERIFY(res.isApprox(res_d) && "full selfadjoint upper to lower");

  res.upperSelfadjointView() = mat.lowerSelfadjointView();
  res_d = lo_sym_d.upperTriangularView();
  VERIFY(res.isApprox(res_d) && "full selfadjoint lower to upper");

  res.lowerSelfadjointView() = mat.lowerSelfadjointView();
  res_d = lo_sym_d.lowerTriangularView();
  VERIFY(res.isApprox(res_d) && "full selfadjoint lower to lower");

  
  
  res.upperSelfadjointView() = mat.upperSelfadjointView().twistedBy(p);
  res_d = ((p * up_sym_d) * p.inverse()).eval().upperTriangularView();
  VERIFY(res.isApprox(res_d) && "full selfadjoint upper twisted to upper");
  
  res.upperSelfadjointView() = mat.lowerSelfadjointView().twistedBy(p);
  res_d = ((p * lo_sym_d) * p.inverse()).eval().upperTriangularView();
  VERIFY(res.isApprox(res_d) && "full selfadjoint lower twisted to upper");
  
  res.lowerSelfadjointView() = mat.lowerSelfadjointView().twistedBy(p);
  res_d = ((p * lo_sym_d) * p.inverse()).eval().lowerTriangularView();
  VERIFY(res.isApprox(res_d) && "full selfadjoint lower twisted to lower");
  
  res.lowerSelfadjointView() = mat.upperSelfadjointView().twistedBy(p);
  res_d = ((p * up_sym_d) * p.inverse()).eval().lowerTriangularView();
  VERIFY(res.isApprox(res_d) && "full selfadjoint upper twisted to lower");
  
  
  res.upperSelfadjointView() = up.upperSelfadjointView().twistedBy(p);
  res_d = ((p * up_sym_d) * p.inverse()).eval().upperTriangularView();
  VERIFY(res.isApprox(res_d) && "upper selfadjoint twisted to upper");
  
  res.upperSelfadjointView() = lo.lowerSelfadjointView().twistedBy(p);
  res_d = ((p * lo_sym_d) * p.inverse()).eval().upperTriangularView();
  VERIFY(res.isApprox(res_d) && "lower selfadjoint twisted to upper");
  
  res.lowerSelfadjointView() = lo.lowerSelfadjointView().twistedBy(p);
  res_d = ((p * lo_sym_d) * p.inverse()).eval().lowerTriangularView();
  VERIFY(res.isApprox(res_d) && "lower selfadjoint twisted to lower");
  
  res.lowerSelfadjointView() = up.upperSelfadjointView().twistedBy(p);
  res_d = ((p * up_sym_d) * p.inverse()).eval().lowerTriangularView();
  VERIFY(res.isApprox(res_d) && "upper selfadjoint twisted to lower");

  
  VERIFY( is_sorted( res = mat.upperSelfadjointView().twistedBy(p) ));
  res_d = (p * up_sym_d) * p.inverse();
  VERIFY(res.isApprox(res_d) && "full selfadjoint upper twisted to full");
  
  VERIFY( is_sorted( res = mat.lowerSelfadjointView().twistedBy(p) ));
  res_d = (p * lo_sym_d) * p.inverse();
  VERIFY(res.isApprox(res_d) && "full selfadjoint lower twisted to full");
  
  VERIFY( is_sorted( res = up.upperSelfadjointView().twistedBy(p) ));
  res_d = (p * up_sym_d) * p.inverse();
  VERIFY(res.isApprox(res_d) && "upper selfadjoint twisted to full");
  
  VERIFY( is_sorted( res = lo.lowerSelfadjointView().twistedBy(p) ));
  res_d = (p * lo_sym_d) * p.inverse();
  VERIFY(res.isApprox(res_d) && "lower selfadjoint twisted to full");
}

template<typename Scalar> void sparse_permutations_all(int size)
{
  CALL_SUBTEST(( sparse_permutations<ColMajor>(SparseMatrix<Scalar, ColMajor>(size,size)) ));
  CALL_SUBTEST(( sparse_permutations<ColMajor>(SparseMatrix<Scalar, RowMajor>(size,size)) ));
  CALL_SUBTEST(( sparse_permutations<RowMajor>(SparseMatrix<Scalar, ColMajor>(size,size)) ));
  CALL_SUBTEST(( sparse_permutations<RowMajor>(SparseMatrix<Scalar, RowMajor>(size,size)) ));
}

EIGEN_DECLARE_TEST(sparse_permutations)
{
  for(int i = 0; i < g_repeat; i++) {
    int s = Eigen::internal::random<int>(1,50);
    CALL_SUBTEST_1((  sparse_permutations_all<double>(s) ));
    CALL_SUBTEST_2((  sparse_permutations_all<std::complex<double> >(s) ));
  }

  VERIFY((internal::is_same<internal::permutation_matrix_product<SparseMatrix<double>,OnTheRight,false,SparseShape>::ReturnType,
                            internal::nested_eval<Product<SparseMatrix<double>,PermutationMatrix<Dynamic,Dynamic>,AliasFreeProduct>,1>::type>::value));

  VERIFY((internal::is_same<internal::permutation_matrix_product<SparseMatrix<double>,OnTheLeft,false,SparseShape>::ReturnType,
                            internal::nested_eval<Product<PermutationMatrix<Dynamic,Dynamic>,SparseMatrix<double>,AliasFreeProduct>,1>::type>::value));
}

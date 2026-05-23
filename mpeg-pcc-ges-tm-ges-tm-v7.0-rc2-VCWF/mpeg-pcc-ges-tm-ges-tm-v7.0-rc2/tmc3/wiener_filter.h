#pragma once

#ifndef wiener_filter_h
#  define wiener_filter_h
#  include <iostream>
#  include <fstream>
#  include <Eigen/Dense>
#  include "AttributeDecoder.h"
#  include "AttributeEncoder.h"
#  include "PCCPointSet.h"
#  define K_wiener 7
#  define K_frame 8

namespace pcc {
using namespace Eigen;
using namespace std;

class wiener {
public:
  wiener(size_t n1, size_t n2, int slide_n);
  wiener();
  void filter(
    PCCPointSet3* recon_pc,
    int qp,
    int slice_n,
    CloudFrame* reconCloud,
    int coef_save[2][K_wiener],
    int coef_Y_save_PCC[7][K_wiener],
    AttributeInterPredParams& attrInterPredParams);
  MatrixXd gbr2yuv(MatrixXd pc_col);
  MatrixXd yuv2gbr(MatrixXd pc_col);
  double cal_psnr(const MatrixXd I, const MatrixXd H);
  double cal_hpsnr(const MatrixXd I, const MatrixXd H);
  bool in_bit;
  void write_bit(
    PCCResidualsEncoder& encoder, CloudFrame* reconCloud, int cw_flag);
  void decode_f(
    PCCPointSet3& recon_pc,
    CloudFrame _outCloud,
    int coef_save_PCC[2][K_wiener],
    int coef_Y_save_PCC[7][K_wiener],
    std::vector<MortonCodeWithIndex> packedVoxel,
    AttributeInterPredParams& attrInterPredParams);
  void read_bit(
    PCCResidualsDecoder& decoder,
    CloudFrame _outCloud,
    AttributeInterPredParams& attrInterPredParams);
  double cost_calculate(size_t bitstream, MatrixXd ori, MatrixXd rec, int qp);
  void round_m(MatrixXd& m);
  void get_o_part(
    size_t p_num,
    PCCPointSet3 pointCloud);
  void get_r_part(
    size_t p_num,
    PCCPointSet3 pointCloud,
    std::vector<MortonCodeWithIndex> packedVoxel,PCCResidualsEncoder& encoder);
  MatrixXd co_part;
  MatrixXd* co_part_Y;
  //MatrixXd co_part_1;
  size_t color_bits;
  size_t o_num;
  size_t p_num;
  bool converted_colors;
  bool wiener_enable_flag;
  bool inheritance_flag;
  bool recolor;
  MatrixXd k_index;
  ~wiener() = default;

private:
  float psnr_ori;
  float psnr_rec;
  int32_t** coef;
  int32_t** coef_y;
  //int32_t* coef_refc;
  int32_t* yuv1;
  int32_t* yuv1_y;
  bool g_lossy;
  bool has_color;
  MatrixXd* cr_part;
  MatrixXd* cr_part_Y;
  //std::vector<int> cr_part_Y_vectors[5];
  std::vector<int> cr_part_Y_vectors[7];
  int partition_n;
  MatrixXd pos;
  
};

//void cal_knn_new(
//  Index** id,
//  const MatrixXd a,
//  const MatrixXd b,
//  const int p_num,
//  const int num,
//  std::vector<int32_t> knn_index);

void cal_knn(
  Index** id,
  const MatrixXd a,
  const MatrixXd b,
  const int p_num,
  const int num);

}  // namespace pcc
#endif /* wiener_filter_h */

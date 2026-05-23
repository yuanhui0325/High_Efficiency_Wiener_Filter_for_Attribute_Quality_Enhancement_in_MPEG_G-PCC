#include <assert.h>
#include <algorithm>
#include <Eigen/Dense>
#include <time.h>
#include <iostream>
#include <fstream>
#include <ctime>
#include <nanoflann.hpp>
#include <unordered_map>
#include "PCCMath.h"
#include "AttributeEncoder.h"
#include "Attribute.h"
#include "AttributeCommon.h"
#include "AttributeDecoder.h"
#include "PCCTMC3Encoder.h"
#include "PCCPointSet.h"
#include "colourspace.h"
#include "PCCTMC3Common.h"
#include "pointset_processing.h"
#include "wiener_filter.h"
#include <bitset>

namespace pcc {
using namespace std;
using namespace Eigen;
using namespace dirac;
class PCCResidualsEncoder : protected AttributeContexts {
public:
  PCCResidualsEncoder(
    const AttributeParameterSet& aps,
    const AttributeBrickHeader& abh,
    const AttributeContexts& ctxtMem);

  EntropyEncoder arithmeticEncoder;

  const AttributeContexts& getCtx() const { return *this; }

  void start(const SequenceParameterSet& sps, int numPoints);
  int sizes();
  int stop();

  void encodeRunLength(int runLength);
  void encodeSymbol(uint32_t value, int k1, int k2, int k3);
  void encode(int32_t value0, int32_t value1, int32_t value2);
  void encode(int32_t value);

  int availPredModes;
  double bitsPtColor(Vec3<int32_t> value, int parity);
  double bitsPtRefl(int32_t value, int parity);

  // Encoder side residual cost calculation
  const int scaleRes = 1 << 20;
  const int windowLog2 = 6;
  int probResGt0[3];  //prob of residuals larger than 0: 1 for each component
  int probResGt1[3];  //prob of residuals larger than 1: 1 for each component
  void resStatUpdateColor(Vec3<int32_t> values);
  void resStatUpdateRefl(int32_t values);
  void resStatReset();
};

class PCCResidualsDecoder : protected AttributeContexts {
public:
  PCCResidualsDecoder(
    const AttributeBrickHeader& abh, const AttributeContexts& ctxtMem);

  EntropyDecoder arithmeticDecoder;

  const AttributeContexts& getCtx() const { return *this; }

  void start(const SequenceParameterSet& sps, const char* buf, int buf_len);
  void stop();

  int decodeRunLength();
  int decodeSymbol(int k1, int k2, int k3);
  void decode(int32_t values[3]);
  int32_t decode();
};

wiener::wiener(size_t n1, size_t n2, int slide_n)
{
  in_bit = false;
  o_num = n1;
  p_num = n2;
  if (n1 > n2)
    g_lossy = true;
  partition_n = slide_n;
  recolor = 0;
}
wiener::wiener()
{
  in_bit = false;
  recolor = 0;
}
void
wiener::filter(
  PCCPointSet3* recon_pc,
  int qp,
  int slice_n,
  CloudFrame* reconCloud,
  int coef_save[3][K_wiener],
  AttributeInterPredParams& attrInterPredParams)
{
  if (!recon_pc->hasReflectances() && !recon_pc->hasColors())
    return;

  //ofstream txt;
  //txt.open("loot_r06.txt", ios::app);
  
  p_num = cr_part[0].rows();

  MatrixXd col_o_yuv(p_num, 3);
  MatrixXd col_r_yuv(p_num, 3);
  col_o_yuv = co_part;
  col_r_yuv.block(0, 0, cr_part[0].rows(), 1) = cr_part[0].col(0);
  col_r_yuv.block(0, 1, cr_part[0].rows(), 1) = cr_part[1].col(0);
  col_r_yuv.block(0, 2, cr_part[0].rows(), 1) = cr_part[2].col(0);

  MatrixXd P(p_num, K_wiener);
  VectorXd S(p_num);
  VectorXd VectorB(K_wiener);
  MatrixXd MatrixA(K_wiener, K_wiener);
  VectorXd Coef(K_wiener);

  yuv1 = new int32_t[3];
  yuv1[0] = 0;
  yuv1[1] = 0;
  yuv1[2] = 0;
  string words = "YUV";

  if (recon_pc->hasColors()) {
    MatrixXd f_pc(p_num, 3);
    MatrixXi f_pc_zheng(p_num, 3);
    for (int i = 0; i < 3; ++i) {
      f_pc.col(i) = col_r_yuv.col(i);
    }

    coef = new int32_t*[K_wiener];
    for (int i = 0; i < K_wiener; ++i) {
      coef[i] = new int32_t[3];
    }

    //cout << "attrInterPredParams.enableAttrInterPred:"
    //     << attrInterPredParams.enableAttrInterPred << endl;

    if (attrInterPredParams.enableAttrInterPred) {
      //if ((reconCloud->frameNum == 0) || (reconCloud->frameNum == 1))
      if (reconCloud->frameNum == 0)
         inheritance_flag = 1;
      else 
         inheritance_flag = 0;
    } else {
      if (reconCloud->frameNum == 0)
         inheritance_flag = 1;
      else
         inheritance_flag = 0;        
    }
    //cout << "inheritance_flag:" << inheritance_flag ;
    //if (attrInterPredParams.enableAttrInterPred) {
    //if (1) {
    //  if ((reconCloud->frameNum) % K_frame == 0)
    //    inheritance_flag = 1;
    //  else
    //    inheritance_flag = 0;
    //} else {
    //  if (reconCloud->frameNum == 0)
    //    inheritance_flag = 1;
    //  else
    //    inheritance_flag = 0;
    //}

    //cout << "inheritance_flag:" << inheritance_flag << endl;
    //if ((reconCloud->frameNum) % K_frame == 0) {
    if (inheritance_flag) {
    //if (1) {
     // txt << col_r_yuv.col(0);
    for (int i = 0; i < 3; ++i) {
      P = cr_part[i];
      S = col_o_yuv.col(i);
      VectorB = P.transpose() * S;
      
      MatrixA = P.transpose() * P;
      
      for (int m = 0; m < K_wiener; ++m) {
         //VectorB(m) = VectorB(m) * 8192;
        VectorB(m) = VectorB(m) * 4096;
        //VectorB(m) = VectorB(m) * 2048;
        //VectorB(m) = VectorB(m) * 1024;
        //VectorB(m) = VectorB(m) * 512;
        //VectorB(m) = VectorB(m) * 256;
      }
      if (isinf((MatrixA.inverse())(0, 0))) {
        for (int i = 0; i < K_wiener; i++) {
          MatrixA(i, i) += 0.0001;
        }
      }
      Coef = MatrixA.inverse() * VectorB;
      //cout << Coef << endl;
      //cout << Coef << endl;
      for (int m = 0; m < K_wiener; ++m) {
         //coef[m][i] = Coef(m) * 10000;
         //Coef(m) = double(coef[m][i]) / 10000.;
         Coef(m) = round(Coef(m));
         coef[m][i] = Coef(m);
         coef_save[i][m] = coef[m][i];   
         //cout << Coef(m) << endl;
      }
      f_pc.col(i) = P * Coef;
      f_pc_zheng = f_pc.cast<int>();
      for (int j = 0; j < p_num; j++) {
         //f_pc_zheng(j, i) = (f_pc_zheng(j, i) + 4096) >> 13;
         f_pc_zheng(j, i) = (f_pc_zheng(j, i) + 2048) >> 12;
         //f_pc_zheng(j, i) = (f_pc_zheng(j, i) + 1024) >> 11;
         //f_pc_zheng(j, i) = (f_pc_zheng(j, i) + 512) >> 10;
          //f_pc_zheng(j, i) = (f_pc_zheng(j, i) + 256) >> 9;
         //f_pc_zheng(j, i) = (f_pc_zheng(j, i) + 128) >> 8;
      }
      f_pc = f_pc_zheng.cast<double>();
      //txt << "reconCloud->frameNum" << reconCloud->frameNum << ": ";
      // txt << words[i] << ": ";
      // for (int ii = 0; ii < K_wiener; ii++)
      //    txt << Coef(ii) << "  ";
      // txt << endl;
    }
    if (converted_colors) {
      round_m(f_pc);
    }
    double r_psnr[3], f_psnr[3], r_J[3], f_J[3];
    for (int i = 0; i < 3; i++) {
      r_psnr[i] = cal_psnr(col_o_yuv.col(i), col_r_yuv.col(i));
      f_psnr[i] = cal_psnr(col_o_yuv.col(i), f_pc.col(i));
      r_J[i] = cost_calculate(
        color_bits / 3., col_o_yuv.col(i), col_r_yuv.col(i), qp);
      //f_J[i] = cost_calculate(
      //  color_bits / 3. +  6 + K_wiener * 2 / K_frame, col_o_yuv.col(i),
      //  f_pc.col(i), qp);
      //f_J[i] = cost_calculate(
      //  color_bits / 3. + 3 + K_wiener * 1 , col_o_yuv.col(i), 
      //  f_pc.col(i), qp);    ////////////
      f_J[i] = cost_calculate(
        color_bits / 3. + 28, col_o_yuv.col(i), f_pc.col(i),
        qp); 
      cout << "f_psnr(" << words[i] << "): " << f_psnr[i] << " , rec_psnr("
           << words[i] << "): " << r_psnr[i] << endl;
      cout << "cost_f(" << words[i] << "): " << f_J[i] << " , cost_r("
           << words[i] << "): " << r_J[i] << endl;

      if (f_psnr[i] > r_psnr[i] && f_J[i] < r_J[i])
       // cout << "1111" << endl;
        yuv1[i] = 1;
    }
    } else {
    //txt << col_r_yuv.col(0);
    for (int i = 0; i < 3; ++i) {
        P = cr_part[i];
      for (int m = 0; m < K_wiener; ++m) {
         //Coef(m) = coef_save[i][m];
         //Coef(m) = double(coef_save[i][m]) / 10000.;
         Coef(m) = double(coef_save[i][m]);
      }
      f_pc.col(i) = P * Coef;
      f_pc_zheng = f_pc.cast<int>();
      for (int j = 0; j < p_num; j++) {
         //f_pc_zheng(j, i) = (f_pc_zheng(j, i) + 4096) >> 13;
          f_pc_zheng(j, i) = (f_pc_zheng(j, i) + 2048) >> 12;
          //f_pc_zheng(j, i) = (f_pc_zheng(j, i) + 1024) >> 11;
         //f_pc_zheng(j, i) = (f_pc_zheng(j, i) + 512) >> 10;
         //f_pc_zheng(j, i) = (f_pc_zheng(j, i) + 256) >> 9;
         //f_pc_zheng(j, i) = (f_pc_zheng(j, i) + 128) >> 8;
      }
      f_pc = f_pc_zheng.cast<double>();
      }
    if (converted_colors) {
         round_m(f_pc);
    }
        
    double r_psnr[3], f_psnr[3], r_J[3], f_J[3];
    for (int i = 0; i < 3; i++) {
        r_psnr[i] = cal_psnr(col_o_yuv.col(i), col_r_yuv.col(i));
        f_psnr[i] = cal_psnr(col_o_yuv.col(i), f_pc.col(i));
        r_J[i] = cost_calculate(
          color_bits / 3., col_o_yuv.col(i), col_r_yuv.col(i), qp);
        //f_J[i] = cost_calculate(
        //  color_bits / 3. + 5 + K_wiener * 2 / K_frame, col_o_yuv.col(i),
        //  f_pc.col(i),
        //  qp);
        //f_J[i] = cost_calculate(
        //  color_bits / 3. + 3 + K_wiener * 1 , col_o_yuv.col(i),
        //  f_pc.col(i), qp);
        f_J[i] = cost_calculate(
          color_bits / 3. + 28, col_o_yuv.col(i), f_pc.col(i),
          qp);
        cout << "f_psnr(" << words[i] << "): " << f_psnr[i] << " , rec_psnr("
             << words[i] << "): " << r_psnr[i] << endl;
        cout << "cost_f(" << words[i] << "): " << f_J[i] << " , cost_r("
             << words[i] << "): " << r_J[i] << endl;

        if (f_psnr[i] > r_psnr[i] && f_J[i] < r_J[i])
           yuv1[i] = 1;
    }
    }

    //if (yuv1[0] == 1 && yuv1[1] == 1 && yuv1[2] == 1) {
    //  //in_bit = true;
    //  cout << "will use wiener filter(YUV) in decoder " << endl;
    //} else if (yuv1[0] == 1 && yuv1[1] == 1 && yuv1[2] == 0) {
    //  //in_bit = true;
    //  cout << "will use wiener filter(YU) in decoder " << endl;
    //  //for (int m = 0; m < K_wiener; ++m) {
    //  //  coef_s[m][2] = 0;
    //  //}
    //} else if (yuv1[0] == 1 && yuv1[1] == 0 && yuv1[2] == 1) {
    //  //in_bit = true;
    //  cout << "will use wiener filter(YV) in decoder " << endl;
    //  //for (int m = 0; m < K_wiener; ++m) {
    //  //  coef_s[m][1] = 0;
    //  //}
    //} else if (yuv1[0] == 0 && yuv1[1] == 0 && yuv1[2] == 1) {
    //  //in_bit = true;
    //  cout << "will use wiener filter(V) in decoder " << endl;
    //  //for (int m = 0; m < K_wiener; ++m) {
    //  //  coef_s[m][0] = 0;
    //  //  coef_s[m][1] = 0;
    //  //}
    //} else if (yuv1[0] == 0 && yuv1[1] == 1 && yuv1[2] == 1) {
    //  //in_bit = true;
    //  cout << "will use wiener filter(UV) in decoder " << endl;
    //  //for (int m = 0; m < K_wiener; ++m) {
    //  //  coef_s[m][0] = 0;
    //  //}
    //} else if (yuv1[0] == 0 && yuv1[1] == 1 && yuv1[2] == 0) {
    //  //in_bit = true;
    //  cout << "will use wiener filter(U) in decoder " << endl;
    //  //for (int m = 0; m < K_wiener; ++m) {
    //  //  coef_s[m][0] = 0;
    //  //  coef_s[m][2] = 0;
    //  //}
    //} else if (yuv1[0] == 1 && yuv1[1] == 0 && yuv1[2] == 0) {
    //  //in_bit = true;
    //  cout << "will use wiener filter(Y) in decoder " << endl;
    //  //for (int m = 0; m < K_wiener; ++m) {
    //  //  coef_s[m][2] = 0;
    //  //  coef_s[m][1] = 0;
    //  //}
    //}
    //if (!in_bit) {
    //  delete[] yuv1;
    //  for (int i = 0; i < K_wiener; i++) {
    //    delete[] coef[i];
    //  }
    //  delete[] coef;
    //  return;
    //}

    for (int j = 0; j < 3; j++) {
      if (yuv1[j] == 0)
      // if (1)
        f_pc.col(j) = col_r_yuv.col(j);
    }

    Vec3<attr_t> col_tmp;
    for (size_t i = 0; i < p_num; i++) {
      for (int j = 0; j < 3; j++) {
        col_tmp[j] = f_pc(i, j);
      }

      recon_pc->setColor(i, col_tmp);
    }

    col_tmp = Vec3<attr_t>();
    S.resize(0);
    Coef.resize(0);
    VectorB.resize(0);
    col_o_yuv.resize(0, 0);
    col_r_yuv.resize(0, 0);
    P.resize(0, 0);
    MatrixA.resize(0, 0);
    f_pc.resize(0, 0);
    co_part.resize(0, 0);
    for (uint8_t i = 0; i < 3; i++) {
      cr_part[i].resize(0, 0);
    }
    delete[] cr_part;
  }
}

void
wiener::decode_f(
  PCCPointSet3& recon_pc,
  CloudFrame _outCloud,
  int coef_save_PCC[3][K_wiener],
  std::vector<MortonCodeWithIndex> packedVoxel,
  AttributeInterPredParams& attrInterPredParams)
{
   MatrixXd coef_d(K_wiener, 3);

  if (!in_bit) { 
    coef_d.setZero();
    //if ((_outCloud.frameNum) % K_frame == 0) {
    //if (1) {
    if (inheritance_flag) {
      for (int i = 0; i < K_wiener; ++i) {
        for (size_t d = 0; d < 3; ++d) {
           coef_d(i, d) = coef[i][d];
           coef_save_PCC[d][i] = coef[i][d];
           //cout << coef[i][d] << endl;
        }
      }
    }
    delete[] yuv1;
    for (int i = 0; i < K_wiener; i++) {
      delete[] coef[i];
    }
    delete[] coef;
    //cout << "coef[i][d]" << endl;
    return;
  }

  size_t p_num = recon_pc.getPointCount();

  has_color = recon_pc.hasColors();

  static const uint16_t kNeighOffset[7] = {7, 3, 5, 6, 35, 21, 14};
  
  if (has_color) {
    //MatrixXd coef_d(K_wiener, 3);
     coef_d.setZero();
   // if ((_outCloud.frameNum) % K_frame == 0) {
   //if (1) {
  //if ((_outCloud.frameNum == 0) || (_outCloud.frameNum == 1)) {
  if (inheritance_flag) {
    for (int i = 0; i < K_wiener; ++i) {
      for (size_t d = 0; d < 3; ++d) {
           //coef_d(i, d) = coef[i][d] / 10000.;
           coef_d(i, d) = coef[i][d];
           coef_save_PCC[d][i] = coef[i][d];
           //cout << coef[i][d] << endl;
        }
      }
   } else {
       for (int i = 0; i < K_wiener; ++i) {
         for (size_t d = 0; d < 3; ++d) {
            //coef_d(i, d) = coef_save_PCC[d][i] / 10000.;
            coef_d(i, d) = coef_save_PCC[d][i];
           //cout << coef_save_PCC[d][i] << endl;
         }
       }
   }
   //cout << yuv1[0] << endl;
   MatrixXd col_r_yuv(p_num, 3);
   for (size_t i = 0; i < p_num; ++i) {
       for (int j = 0; j < 3; ++j) {
          col_r_yuv(i, j) = recon_pc.getColor(i)[j];
       }
   }

   MatrixXd PY(p_num, K_wiener);
   MatrixXd PU(p_num, K_wiener);
   MatrixXd PV(p_num, K_wiener);
   MatrixXd f_pc = col_r_yuv;
   MatrixXi f_zheng(p_num, 3);

    MortonIndexMap3d atlas;
    atlas.resize(7);
    atlas.init();
    atlas.reserve(p_num);
    int64_t curAtlasId = -1;
    int64_t cubeIndex = 0;
    int64_t lastMortonCodeShift3 = -1;
    for (int n = 0; n < p_num; n++) {
      //int count = 1;
      const auto& pv = packedVoxel[n];
      const int64_t mortonCode = pv.mortonCode;
      const int64_t pointAtlasId = mortonCode >> 21;
      const int64_t mortonCodeShiftBits3 = mortonCode;
      if (curAtlasId != pointAtlasId) {
        atlas.clearUpdates();
        curAtlasId = pointAtlasId;
        while (cubeIndex < p_num
               && (packedVoxel[cubeIndex].mortonCode >> 21) == curAtlasId) {
           atlas.set(
             packedVoxel[cubeIndex].mortonCode, packedVoxel[cubeIndex].index);
           ++cubeIndex;
        }
      }
      if (lastMortonCodeShift3 != mortonCodeShiftBits3) {
        lastMortonCodeShift3 = mortonCodeShiftBits3;
        const auto basePosition = morton3dAdd(mortonCodeShiftBits3, -1ll);
        if (yuv1[0] == 1)
           PY(pv.index, 0) = col_r_yuv(pv.index, 0);
        if (yuv1[1] == 1)
           PU(pv.index, 0) = col_r_yuv(pv.index, 1);
        if (yuv1[2] == 1)
           PV(pv.index, 0) = col_r_yuv(pv.index, 2);
        //for (int32_t m = 1; m < K_wiener; ++m) {
        for (int32_t m = 1; m < K_wiener; ++m) {
           const auto neighbMortonCode =
             morton3dAdd(basePosition, kNeighOffset[m]);
           if ((neighbMortonCode >> 21) != curAtlasId) {  //21
          if (yuv1[0] == 1)
            PY(pv.index, m) = col_r_yuv(pv.index, 0);
          if (yuv1[1] == 1)
            PU(pv.index, m) = col_r_yuv(pv.index, 1);
          if (yuv1[2] == 1)
            PV(pv.index, m) = col_r_yuv(pv.index, 2);
          continue;
           }
           const auto range = atlas.get(neighbMortonCode);
           if (range.start != -1) {
          if (yuv1[0] == 1)
            PY(pv.index, m) = col_r_yuv(range.start, 0);
          if (yuv1[1] == 1)
            PU(pv.index, m) = col_r_yuv(range.start, 1);
          if (yuv1[2] == 1)
            PV(pv.index, m) = col_r_yuv(range.start, 2);
          //if (yuv1[0] == 1)
          //  PY(pv.index, count) = col_r_yuv(range.start, 0);
          //if (yuv1[1] == 1)
          //  PU(pv.index, count) = col_r_yuv(range.start, 1);
          //if (yuv1[2] == 1)
          //  PV(pv.index, count) = col_r_yuv(range.start, 2);
          // count = count + 1;
           } else {
          if (yuv1[0] == 1)
            PY(pv.index, m) = col_r_yuv(pv.index, 0);
          if (yuv1[1] == 1)
            PU(pv.index, m) = col_r_yuv(pv.index, 1);
          if (yuv1[2] == 1)
            PV(pv.index, m) = col_r_yuv(pv.index, 2);
           }
        }
      }
      //for (int m = count; m < K_wiener; m++) {
      //  if (yuv1[0] == 1)
      //     PY(pv.index, m) = col_r_yuv(pv.index, 0);
      //  if (yuv1[1] == 1)
      //     PU(pv.index, m) = col_r_yuv(pv.index, 1);
      //  if (yuv1[2] == 1)
      //     PV(pv.index, m) = col_r_yuv(pv.index, 2);
      //}
    }
    atlas.resize(0);
    atlas.reserve(0);

   //////////////////////////
    //MatrixXd pos_r(p_num, 3);
    //for (size_t i = 0; i < p_num; ++i) {
    //  for (int j = 0; j < 3; ++j) {
    //    pos_r(i, j) = recon_pc[i][j];
    //  }
    //}
    //Index** index_id;
    //;
    //index_id = new Index*[p_num];
    //for (size_t i = 0; i < p_num; ++i) {
    //  index_id[i] = new Index[K_wiener];
    //}
    //cal_knn(index_id, pos_r, pos_r, p_num, K_wiener);
    //for (size_t i = 0; i < p_num; ++i) {
    //  delete[] index_id[i];
    //}
    //delete[] index_id;
    ///////////////////////////////

     if (yuv1[0] == 1) {
        //f_pc.col(0) = PY * coef_d.col(0);
       f_pc.col(0) = PY * coef_d.col(0);
       f_zheng.col(0) = f_pc.col(0).cast<int>();
       for (int j = 0; j < p_num; j++) {
        //f_zheng(j, 0) = (f_zheng(j, 0) + 4096) >> 13;
         f_zheng(j, 0) = (f_zheng(j, 0) + 2048) >> 12;
        //f_zheng(j, 0) = (f_zheng(j, 0) + 1024) >> 11;
        //f_zheng(j, 0) = (f_zheng(j, 0) + 512) >> 10;
        //f_zheng(j, 0) = (f_zheng(j, 0) + 256) >> 9;
        //f_zheng(j, 0) = (f_zheng(j, 0) + 128) >> 8;
         if (f_zheng(j, 0) < 0) {
            f_zheng(j, 0) = 0;
         }
         if (f_zheng(j, 0) > 255) {
            f_zheng(j, 0) = 255;
         }
       }
       f_pc.col(0) = f_zheng.col(0).cast<double>();
     } else {
        f_pc.col(0) = col_r_yuv.col(0);
     }
     if (yuv1[1] == 1) {
        f_pc.col(1) = PU * coef_d.col(1);
        f_zheng.col(1) = f_pc.col(1).cast<int>();
        for (int j = 0; j < p_num; j++) {
         //f_zheng(j, 1) = (f_zheng(j, 1) + 4096) >> 13;
         f_zheng(j, 1) = (f_zheng(j, 1) + 2048) >> 12;
         //f_zheng(j, 1) = (f_zheng(j, 1) + 1024) >> 11;
         // f_zheng(j, 1) = (f_zheng(j, 1) + 512) >> 10;
         //f_zheng(j, 1) = (f_zheng(j, 1) + 256) >> 9;
         //f_zheng(j, 1) = (f_zheng(j, 1) + 128) >> 8;
         if (f_zheng(j, 1) < 0) {
            f_zheng(j, 1) = 0;
         }
         if (f_zheng(j, 1) > 255) {
            f_zheng(j, 1) = 255;
         }
        }
        f_pc.col(1) = f_zheng.col(1).cast<double>();
     } else {
        f_pc.col(1) = col_r_yuv.col(1);
     }
     if (yuv1[2] == 1) {
        f_pc.col(2) = PV * coef_d.col(2);
        f_zheng.col(2) = f_pc.col(2).cast<int>();
        for (int j = 0; j < p_num; j++) {
         //f_zheng(j, 2) = (f_zheng(j, 2) + 4096) >> 13;
         f_zheng(j, 2) = (f_zheng(j, 2) + 2048) >> 12;
         //f_zheng(j, 2) = (f_zheng(j, 2) + 1024) >> 11;
         //f_zheng(j, 2) = (f_zheng(j, 2) + 512) >> 10;
         //f_zheng(j, 2) = (f_zheng(j, 2) + 256) >> 9;
           //f_zheng(j, 2) = (f_zheng(j, 2) + 128) >> 8;
         if (f_zheng(j, 2) < 0) {
            f_zheng(j, 2) = 0;
         }
         if (f_zheng(j, 2) > 255) {
            f_zheng(j, 2) = 255;
         }
        }
        f_pc.col(2) = f_zheng.col(2).cast<double>();
     } else {
        f_pc.col(2) = col_r_yuv.col(2);
     }
    //round_m(f_pc);

    for (size_t i = 0; i < p_num; ++i) {
      for (int j = 0; j < 3; j++) {
        (recon_pc.getColor(i))[j] = f_pc(i, j);
      }
    }
    delete[] yuv1;
    for (int i = 0; i < K_wiener; i++) {
       delete[] coef[i];
    }
    delete[] coef;
    
    coef_d.resize(0, 0);
    col_r_yuv.resize(0, 0);
    PY.resize(0, 0);
    PU.resize(0, 0);
    PV.resize(0, 0);
    f_pc.resize(0, 0);
  }

}

void
wiener::write_bit(
  PCCResidualsEncoder& encoder,
  CloudFrame* reconCloud,
  int cw_flag)
{
  //if (yuv1[0] == 1 || yuv1[1] == 1 || yuv1[2] == 1) {
    //uint32_t acDataLen = encoder.sizes();
    //color_bits = acDataLen * 8;
    //cout << color_bits << endl;
  if ((reconCloud->frameNum) == 0) {
    if (cw_flag == 0)
      encoder.arithmeticEncoder.encode(0);
    else
      encoder.arithmeticEncoder.encode(1);
     //encoder.arithmeticEncoder.encode(0);
     //encoder.arithmeticEncoder.encode(0);
  }
      //encoder.encode(2, 2, 2);
    //acDataLen = encoder.sizes();
    //color_bits = acDataLen * 8;
    //cout << color_bits << endl;
   //encoder.arithmeticEncoder.encode(2);
  //cout << yuv1[0] << endl;
  //cout << yuv1[1] << endl;
  //cout << yuv1[2] << endl;

  //if (in_bit)
    encoder.arithmeticEncoder.encode(yuv1[0]);
    encoder.arithmeticEncoder.encode(yuv1[1]);
    encoder.arithmeticEncoder.encode(yuv1[2]);
    //encoder.encode(yuv1[0], yuv1[1], yuv1[2]);
    //acDataLen = encoder.sizes();
    //color_bits = acDataLen * 8;
    //cout << color_bits << endl;
    uint32_t acDataLen;
    //if ((reconCloud->frameNum) % K_frame == 0) {
    //cout << "inheritance_flag22:" << inheritance_flag << endl;

    if (inheritance_flag) {
    //if (1) {
    for (int i = 0; i < K_wiener; ++i) {
         //cout << coef[i][0] << endl;
         //cout << coef[i][1] << endl;
         //cout << coef[i][2] << endl;
         //assert(IntToUInt(coef[i][1]) < std::numeric_limits<uint32_t>::max());
         //acDataLen = encoder.sizes();
         //color_bits = acDataLen * 8;
         //cout << color_bits << endl;
         encoder.encode(coef[i][0], coef[i][1], coef[i][2]);
         //acDataLen = encoder.sizes();
         //color_bits = acDataLen * 8;
         //cout << color_bits << endl;
      }
    //acDataLen = encoder.sizes();
    //  color_bits = acDataLen * 8;
    //  cout << color_bits << endl;
    }
  //} 
   //cout << "yuv1[2]" << endl;
  delete[] yuv1;
    //cout << "yuv1[2]" << endl;
  for (int i = 0; i < K_wiener; i++) {
      delete[] coef[i];
  }
  //cout << "yuv1[2]" << endl;
  delete[] coef;
  //cout << "yuv1[2]" << endl;
}

void
wiener::read_bit(
  PCCResidualsDecoder& decoder,
  CloudFrame _outCloud,
  AttributeInterPredParams& attrInterPredParams)
{
  
  yuv1 = new int32_t[3];

  if ((_outCloud.frameNum) == 0) {
   recolor = 1;
   int32_t wiener_flag;
  // decoder.decode(pre);
   wiener_flag = decoder.arithmeticDecoder.decode();
   //cout << pre[0] << endl;
   //cout << pre[1] << endl;
   //cout << pre[2] << endl;
   if (wiener_flag == 0) {
      recolor = 0;
  //  in_bit = true;
  //  has_color = true;
      } 
    }
  if (recolor)
      return;

  if (attrInterPredParams.enableAttrInterPred) {
      //if ((_outCloud.frameNum == 0) || (_outCloud.frameNum == 1))
    if (_outCloud.frameNum == 0)
      inheritance_flag = 1;
      else
      inheritance_flag = 0;
  } else {
      if (_outCloud.frameNum == 0)
      inheritance_flag = 1;
      else
      inheritance_flag = 0;
  }
  
  //if (attrInterPredParams.enableAttrInterPred) {
  //if (1) {
  //    if ((_outCloud.frameNum) % K_frame == 0)
  //    inheritance_flag = 1;
  //    else
  //    inheritance_flag = 0;
  //} else {
  //    if (_outCloud.frameNum == 0)
  //    inheritance_flag = 1;
  //    else
  //    inheritance_flag = 0;
  //}

  //if (!in_bit) {
  //  delete[] yuv1;
  //  return;
  //}
    //decoder.decode(yuv1);
    yuv1[0] = decoder.arithmeticDecoder.decode();
    yuv1[1] = decoder.arithmeticDecoder.decode();
    yuv1[2] = decoder.arithmeticDecoder.decode();
    //cout << yuv1[0] << endl;
    //cout << yuv1[0] << endl;
    //cout << yuv1[0] << endl;
    if (yuv1[0] || yuv1[1] || yuv1[2])
      in_bit = true;
    //cout << in_bit << endl;
    coef = new int32_t*[K_wiener];
    for (int i = 0; i < K_wiener; i++) {
    coef[i] = new int32_t[3];
    }
    //if ((_outCloud.frameNum) % K_frame == 0) {
    //if (1) {
    //if ((_outCloud.frameNum == 0) || (_outCloud.frameNum == 1)) {
  //cout << "inheritance_flag22:" << inheritance_flag << endl;
  if (inheritance_flag) {
    for (int i = 0; i < K_wiener; i++) {
       int32_t c1[3];
       decoder.decode(c1);
       for (size_t d = 0; d < 3; ++d) {
          coef[i][d] = c1[d];
          //cout << coef[i][d] << endl;
       }
    }
  }
}

MatrixXd
wiener::gbr2yuv(MatrixXd pc_col)
{
  size_t p_num = pc_col.rows();
  RowVectorXd bias_1(1);
  /*bias_1 << 128 * (255 / 256.);*/
  bias_1 << 128;
  MatrixXd bias(p_num, 1);
  bias = MatrixXd::Zero(p_num, 1);
  bias.rowwise() += bias_1;

  MatrixXd r = pc_col.block(0, 2, p_num, 1);
  MatrixXd g = pc_col.block(0, 0, p_num, 1);
  MatrixXd b = pc_col.block(0, 1, p_num, 1);
  MatrixXd color_yuv(p_num, 3);
  //convert yuv
  color_yuv.block(0, 0, p_num, 1) = 0.212600 * r + 0.715200 * g + 0.072200 * b;
  color_yuv.block(0, 1, p_num, 1) =
    -0.114572 * r - 0.385428 * g + 0.5 * b + bias;
  color_yuv.block(0, 2, p_num, 1) =
    0.5 * r - 0.454153 * g - 0.045847 * b + bias;
  return color_yuv;
}

MatrixXd
wiener::yuv2gbr(MatrixXd pc_col)
{
  size_t p_num = pc_col.rows();
  RowVectorXd bias_1(1);
  bias_1 << 128;
  MatrixXd bias(p_num, 1);
  bias.setZero();
  bias.rowwise() += bias_1;

  MatrixXd y = pc_col.block(0, 0, p_num, 1);
  MatrixXd u = pc_col.block(0, 1, p_num, 1) - bias;
  MatrixXd v = pc_col.block(0, 2, p_num, 1) - bias;
  MatrixXd color_gbr(p_num, 3);
  //convert yuv
  color_gbr.block(0, 0, p_num, 1) = y - 0.18733 * u - 0.46813 * v;
  color_gbr.block(0, 1, p_num, 1) = y + 1.85563 * u /*+ 0.00000 * v*/;
  color_gbr.block(0, 2, p_num, 1) = y /*- 0.00000 * u*/ + 1.57480 * v;
  return color_gbr;
}

void
wiener::get_o_part(
  size_t p_num,
  PCCPointSet3 pointCloud)
{
  co_part.resize(p_num, 3);
  for (int n = 0; n < p_num; n++) {
     for (int j = 0; j < 3; j++) {
        co_part(n, j) = pointCloud.getColor(n)[j];
     }
  }
  //cout << " rec:"  << endl;
  /////////////////////////////////
  //MatrixXd pos_o(p_num, 3), pos_r(p_num, 3);
  //Index** o2r;
  //o2r = new Index*[p_num];
  //for (size_t i = 0; i < p_num; i++) {
  //   o2r[i] = new Index[1];
  //   for (int j = 0; j < 3; j++) {
  //      pos_o(i, j) = pointCloud[i][j];
  //      pos_r(i, j) = pointCloud[i][j];
  //   }
  //}
  //cal_knn(o2r, pos_o, pos_r, p_num, 1);
  //for (size_t i = 0; i < p_num; i++) {
  //   delete[] o2r[i];
  //}
  //delete[] o2r;
  /////////////////////////////////////
  //cout << " ori:" << pointCloud[0] << endl;
}

void
wiener::get_r_part(
  size_t p_num,
  PCCPointSet3 pointCloud,
  std::vector<MortonCodeWithIndex> packedVoxel,
  PCCResidualsEncoder& encoder)
{
  //cout << " rec:" << pointCloud[0] << endl;
  uint32_t acDataLen = encoder.sizes();
  color_bits = acDataLen * 8;
  //cout << color_bits << endl;
  //clock_t st, ed, st1, ed1, st2, ed2, st3, ed3;
  cr_part = new MatrixXd[3];
  cr_part[0].resize(pointCloud.getPointCount(), K_wiener);
  cr_part[1].resize(pointCloud.getPointCount(), K_wiener);
  cr_part[2].resize(pointCloud.getPointCount(), K_wiener);

  PCCPointSet3* P_tmp = &pointCloud;

   static const uint16_t kNeighOffset[7] = {
      7,  3,  5,  6,  35,  21,  14};


   MortonIndexMap3d atlas;
   atlas.resize(9);
   atlas.init();
   atlas.reserve(p_num);

   int64_t curAtlasId = -1;
   int64_t cubeIndex = 0;
   int64_t lastMortonCodeShift3 = -1;

   std::bitset<64> binary(kNeighOffset[0]);  
   std::cout << binary << std::endl;

   for (int n = 0; n < p_num; n++) {
       //int count = 1;
       const auto& pv = packedVoxel[n];
       //if (n == 0)
         //cout << " rec:" << pointCloud[pv.index] << endl;
       const int64_t mortonCode = pv.mortonCode;
       const int64_t pointAtlasId = mortonCode >> 27; // 27
       //cout << " mortonCode:" << mortonCode << endl;
       //cout << " pointAtlasId:" << pointAtlasId << endl;
       const int64_t mortonCodeShiftBits3 = mortonCode;
       if (curAtlasId != pointAtlasId) {
        //cout << " curAtlasId:" << curAtlasId << endl;
        atlas.clearUpdates();
        curAtlasId = pointAtlasId;
          while (
              cubeIndex < p_num
               && (packedVoxel[cubeIndex].mortonCode >> 27)
                 == curAtlasId) {
              atlas.set(
             packedVoxel[cubeIndex].mortonCode, packedVoxel[cubeIndex].index);
              ++cubeIndex;
          }
       }
       if (lastMortonCodeShift3 != mortonCodeShiftBits3) {
          lastMortonCodeShift3 = mortonCodeShiftBits3;
          const auto basePosition = morton3dAdd(mortonCodeShiftBits3, -1ll);
          //const auto basePosition2 =
          //  morton3dminus(mortonCodeShiftBits3, kNeighOffset[0]);
          //cout << " basePosition:" << basePosition << endl;
          //cout << " basePosition2:" << basePosition2 << endl;
          //cout << " mortonCodeShiftBits3:" << mortonCodeShiftBits3 << endl;
          cr_part[0](pv.index, 0) =
          (*P_tmp).getColor(pv.index)[0];
          cr_part[1](pv.index, 0) =
            (*P_tmp).getColor(pv.index)[1];
          cr_part[2](pv.index, 0) =
            (*P_tmp).getColor(pv.index)[2];
          //for (int32_t m = 1; m < K_wiener; ++m) {
          for (int32_t m = 1; m < K_wiener; ++m) {
              const auto neighbMortonCode =
                morton3dAdd(basePosition, kNeighOffset[m]);
              if (m == 2) {
              const auto neighbMortonCode2 = morton3dAdd(
              mortonCodeShiftBits3, 0x2492492492492492llu);
              cout << " neighbMortonCode:" << neighbMortonCode << endl;
              cout << " neighbMortonCode2:" << neighbMortonCode2 << endl;
              }
              if ((neighbMortonCode >> 27) != curAtlasId) {  //21
            cr_part[0](pv.index, m) =
              (*P_tmp).getColor(pv.index)[0];
            cr_part[1](pv.index, m) =
              (*P_tmp).getColor(pv.index)[1];
            cr_part[2](pv.index, m) =
              (*P_tmp).getColor(pv.index)[2];
                 continue;
              }
              const auto range = atlas.get(neighbMortonCode);
              if (range.start != -1) {
            cr_part[0](pv.index, m) = (*P_tmp).getColor(range.start)[0];
            cr_part[1](pv.index, m) = (*P_tmp).getColor(range.start)[1];
            cr_part[2](pv.index, m) = (*P_tmp).getColor(range.start)[2];
                 //cr_part[0](pv.index, count) =
                 //  (*P_tmp).getColor(range.start)[0];
                 //cr_part[1](pv.index, count) =
                 //  (*P_tmp).getColor(range.start)[1];
                 //cr_part[2](pv.index, count) =
                 //  (*P_tmp).getColor(range.start)[2];
                 //count = count + 1;
              } 
           //if (count == K_wiener)
           //      break;
           else {
            cr_part[0](pv.index, m) = (*P_tmp).getColor(pv.index)[0];
            cr_part[1](pv.index, m) = (*P_tmp).getColor(pv.index)[1];
            cr_part[2](pv.index, m) = (*P_tmp).getColor(pv.index)[2];
              }
          }
       }
       //for (int m = count; m < K_wiener; m++) {
       //   cr_part[0](pv.index, m) = (*P_tmp).getColor(pv.index)[0];
       //   cr_part[1](pv.index, m) = (*P_tmp).getColor(pv.index)[1];
       //   cr_part[2](pv.index, m) = (*P_tmp).getColor(pv.index)[2];
       //}
       //cout << count << endl;
   }
   atlas.resize(0);
   atlas.reserve(0);
   ///////////////////
   //MatrixXd pos_r(p_num, 3);
   //Index** index_knn;
   //index_knn = new Index*[p_num];
   //for (size_t i = 0; i < p_num; i++) {
   //    index_knn[i] = new Index[K_wiener];
   //    for (int j = 0; j < 3; j++) {
   //       pos_r(i, j) = pointCloud[i][j];
   //    }
   //}
   //cal_knn(index_knn, pos_r, pos_r, p_num, K_wiener);
   //for (size_t i = 0; i < p_num; i++) {
   //    delete[] index_knn[i];
   //}
   //delete[] index_knn;
   ////////////////////////
}

double
wiener::cal_psnr(const MatrixXd I, const MatrixXd H)
{
  MatrixXd tmp = (I - H).cwiseProduct(I - H);
  double mse = tmp.sum() / tmp.size();
  //cout << mse << endl;
  double psnr = 10 * log10(255 * 255 / mse);
  return psnr;
}

double
wiener::cal_hpsnr(const MatrixXd I, const MatrixXd H)
{
  MatrixXd tmp = (I - H);
  double mse = tmp.maxCoeff() * tmp.maxCoeff();
  //cout << mse << endl;
  double psnr = 10 * log10(255 * 255 / mse);
  return psnr;
}

double
wiener::cost_calculate(size_t bitstream, MatrixXd ori, MatrixXd rec, int qp)
{
  double J = 0;
  double lamda = 0;
  double dist = 0;
  /*int row = ori.rows();
  int col = ori.cols();
  for (int r = 0; r < row; r++) {
    for (int co = 0; co < col; co++) {
      dist += pow(ori(r, co) - rec(r, co), 2);
    }
  }*/
  auto temp = (ori - rec).cwiseProduct(ori - rec);
  dist = temp.sum();
  lamda = 0.85 * pow(2, (qp - 12) / 3.0);
  J = dist + lamda * bitstream;
  return J;
}


//void
//cal_knn_new(
//  Index** id,
//  //source, tree
//  const MatrixXd a,
//  //target
//  const MatrixXd b,
//  const int p_num,
//  const int num,
//  std::vector<int32_t> knn_index)
//{
//  //clock_t st_knn1, ed_knn2;
//  //st_knn1 = clock();
//  //cal_knn(index_id, pos_r, pos_r, p_num, K_wiener);
//  //a=o_pc.location,b=r_pc.location
//  MatrixXd b_t = b.transpose();
//
//  nanoflann::KDTreeEigenMatrixAdaptor<MatrixXd> tree1(a, 15);
//  tree1.index->buildIndex();
//  double** dists_id;
//  dists_id = new double*[p_num];
//  for (size_t i = 0; i < p_num; ++i) {
//    dists_id[i] = new double[num];
//  }
//  size_t nMatches1;
//
//  for (size_t i = 0; i < p_num; ++i) {
//    //nMatches1 = tree1.index->knnSearch(&b_t(0, i), num, id[i], dists_id[i]);
//    nMatches1 =
//      tree1.index->knnSearch(&b_t(0, i), num, id[knn_index[i]], dists_id[i]);
//  }
//  for (size_t i = 0; i < p_num; ++i) {
//    delete[] dists_id[i];
//  }
//  delete[] dists_id;
//  return ;
//}

void
cal_knn(
  Index** id,
  //source, tree
  const MatrixXd a,
  //target
  const MatrixXd b,
  const int p_num,
  const int num)
{
  //a=o_pc.location,b=r_pc.location
  MatrixXd b_t = b.transpose();

  nanoflann::KDTreeEigenMatrixAdaptor<MatrixXd> tree1(a, 15);
  tree1.index->buildIndex();

  double** dists_id;
  dists_id = new double*[p_num];
  for (size_t i = 0; i < p_num; ++i) {
    dists_id[i] = new double[num];
  }
  size_t nMatches1;

  for (size_t i = 0; i < p_num; ++i) {
    nMatches1 = tree1.index->knnSearch(&b_t(0, i), num, id[i], dists_id[i]);
  }
  for (size_t i = 0; i < p_num; ++i) {
    delete[] dists_id[i];
  }
  delete[] dists_id;
  return;
}

void
wiener::round_m(MatrixXd& m)
{
  for (int i = 0; i < m.rows(); ++i) {
    for (int j = 0; j < m.cols(); ++j) {
      m(i, j) = round(m(i, j));
      if (m(i, j) < 0)
        m(i, j) = 0;
      if (m(i, j) > 255)
        m(i, j) = 255;
    }
  }
}

}  // namespace pcc
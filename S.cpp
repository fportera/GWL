#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <unistd.h>

#define NFEAT 100
#define NTRAIN 2000

#define MAX(a, b) ((a) < (b) ? (b) : (a))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define FMIN(a, b) ((fabs(a)) < (fabs(b)) ? (a) : (b))

char dsname[1000];
int nFeat;

int trsize, tesize;

double X[NTRAIN][NFEAT];
double y[NTRAIN];
double X_train[NTRAIN][NFEAT];
double X_test[NTRAIN];
double y_train[NTRAIN][NFEAT];
double y_test[NTRAIN];
double K[NTRAIN][NTRAIN];
double S[NTRAIN][NTRAIN];
double s[NTRAIN];

double gammaK;
double gammaS;
double C;

double a[NTRAIN];
double b, pb;
double ba[NTRAIN];
double bb, bpb;

double dof = 0;

double f[NTRAIN];
double bf[NTRAIN];
double xi[NTRAIN];

int cases =0;

double totBdof = 0;
int nDuals = 0;
int patience = 0;
double tot_s = 0;

void LoadAndPrepareData(int fold) { 
  char buffer[10000];
  FILE *f;

  f = fopen(dsname, "r");

  int ltot = 0;

  while(fscanf(f, "%s\n", buffer) != EOF) {
    int i = 0;
    char *pt;
    
    pt = strtok (buffer, ",");
    while (pt != NULL) {
        double a = atof(pt);
	if (i < nFeat) { // skip the code
	  X[ltot][i]= a;
	} else
	  if (i==nFeat) {
	    y[ltot] = (int)a;
	  }
	pt = strtok(NULL, ",");
	i++;
    }
    ltot++;
  }
  fclose(f);

  // printf("X:");
  // for(int i = 0; i < ltot; i++) {
  //   for(int j = 0; j < nFeat; j++) {
  //     printf("%lf,", X[i][j]);
  //   }
  //   printf("\n");
  // }
  // printf("y:");
  // for(int i = 0; i < ltot; i++) {
  //     printf("%lf,", y[i]);
  // }
  // printf("\n");
  
  trsize = (int)(4 * ltot / 5);
  tesize = int(ltot / 5);

  printf("trsize = %d\n", trsize);
  printf("tesize = %d\n", tesize);
  
  int cTr = 0;
  int cTe = 0;
  
  for(int i = 0; i < ltot; i++) {
    if (!((fold * tesize <= i) && (i < (fold+1) * tesize))) {
      for(int j = 0; j < nFeat; j++) {
	X_train[cTr][j] = X[i][j];
      }
      X_test[cTr] = y[i];
      cTr++;
    } else {
      for(int j = 0; j < nFeat; j++)
	y_train[cTe][j] = X[i][j];
      y_test[cTe] = y[i];
      // printf("cTe = %d\n", cTe);
      cTe++;
    }
  }

  printf("cTr = %d\n", cTr);
  printf("cTe = %d\n", cTe);
  trsize = cTr;
  tesize = cTe;
  
}

double trDot(int p1, int p2) {

  double d = 0;
  
  for(int i = 0 ; i < nFeat; i++) {
    d += X_train[p1][i] * X_train[p2][i];
  }

  return d;
}

double trteDot(int p1, int p2) {

  double d = 0;
  
  for(int i = 0 ; i < nFeat; i++) {
    d += X_train[p1][i] * y_train[p2][i];
  }

  return d;
}

double teteDot(int p1, int p2) {

  double d = 0;
  
  for(int i = 0 ; i < nFeat; i++) {
    d += y_train[p1][i] * y_train[p2][i];
  }

  return d;
}

double KernelK(int p1, int p2) {
  return exp( -gammaK * (trDot(p1, p1) - 2 * trDot(p1, p2) + trDot(p2, p2)));
}

double KernelS(int p1, int p2) {
  return exp( -gammaS * (trDot(p1, p1) - 2 * trDot(p1, p2) + trDot(p2, p2)));
}

void DetermineK() {
  for(int j = 0; j < trsize; j++) {
    for(int k = 0; k < trsize; k++) {
      K[j][k] = KernelK(j, k);
    }
  }
}

void Determine_s() {
  for(int j = 0; j < trsize; j++) {
    s[j] = 0;
    for(int k = 0; k < trsize; k++) {
      S[j][k] = KernelS(j, k);
      s[j] += S[j][k];
    }
    // s[j] = sqrt(s[j]) ;      // 1
    //s[j] = s[j];              // 2
    //s[j] = s[j] * s[j];       // 3
    // s[j] = 1/sqrt(s[j]) ;    // 4
    // s[j] = 1/s[j];           // 5
    // s[j] = 1/(s[j] * s[j]);  // 6
    // s[j] = s[j] * X_test[j]; // 7
    s[j] = 1 + rand() / (double) RAND_MAX;    // 8
    
    tot_s += s[j];
    //    printf("s[%d]= %lf\n", j, s[j]);
  }
}

double POF() {

  double pof = 0;

  int l = trsize;
  
  for(int j = 0; j < l; j++) {
    f[j] = b;
    for(int k = 0; k < l; k++) {
      f[j] += a[k] * X_test[k] * K[k][j];
    }
  }
  
  for(int j = 0; j < l; j++) {
    xi[j] = MAX(0, 1 - X_test[j] * f[j]);
  }

  for(int j = 0; j < l; j++) {
    for(int k = 0; k < l; k++) {
      pof += a[j] * X_test[j] * a[k] * X_test[k] * K[j][k];
    }
  }
  pof *= 0.5;

  double err = 0;
  for(int k = 0; k < l; k++)
    err += C * xi[k] * s[k];
  
  //  printf("REGULARIZATION = %lf, C * SAMPLES ERROR = %lf\n", pof, err);

  pof = pof + err;
  
  return pof;
}

double DOF() {

  double dof = 0;

  int l = trsize;
  
  for(int j = 0; j < l; j++) {
    dof += a[j];
  }

  double suma = dof;
  
  for(int j = 0; j < l; j++) {
    for(int k = 0; k < l; k++) {
      dof -= 0.5 * a[j] * X_test[j] * a[k] * X_test[k] * K[j][k];
    }
  }
  //  printf("sum a = %lf, - 0.5 sum sum = %lf\n", suma, dof -suma);
  
  return dof;
}


void Update_f(int p, int q, double pap, double paq, double nap, double naq, double pb, double nb) {
  for(int i = 0; i < trsize; i++) {
    f[i] += - pap * X_test[p] * K[i][p] - paq * X_test[q] * K[i][q] + nap * X_test[p] * K[i][p] + naq * X_test[q] * K[i][q] - pb + nb;
  }
}

double clip(double onu, double nai, double naj, int i, int j, int nIter) {
  double nu = 0;

  cases = -2;
  double v1, v2;
  //  printf("UNCLIPPED nu = %lf\n", onu);
  if ((nai < 0) || (nai > C * s[i]) || (naj < 0) || (naj > C * s[j])) {   // vars constraints violation
    if ((X_test[i] == 1) && (X_test[j] == 1)) {
      cases = 0;
      v1 = MAX(- a[i], a[j] - C * s[j]);
      v2 = MIN(  a[j], C * s[i] - a[i]);
      if (v1 > v2)
	nu = 0;
      else
	nu = (v1+v2)/2;
      //nu = MAX(v2, v1);
      // printf("v1 = %lf, v2 = %lf\n", v1, v2);
    } else
      if ((X_test[i] == 1) && (X_test[j] == -1)) {
	cases = 1;
	v1 = MAX(-a[i], -a[j]);
	v2 = MIN(C * s[i] - a[i], C * s[j] - a[j]);
	if (v1 > v2)
	  nu = 0;
	else
	  nu =  (v1+v2)/2;
	//nu = MAX(v2, v1);
	// printf("v1 = %lf, v2 = %lf\n", v1, v2);
      } else
	if ((X_test[i] == -1) && (X_test[j] == 1)) {
	  cases = 2;
	  v1 = MIN(a[i], a[j]);
	  v2 = MAX(a[i] - C * s[i], a[j] - C * s[j]);
	  if (v2 > v1)
	    nu = 0;
	  else
	    //		    nu = (rand()/((double)RAND_MAX)*(v1-v2) + v1;
	    nu = (v1+v2)/2;
	    //  nu = MAX(v2, v1);
	  // printf("v1 = %lf, v2 = %lf\n", v1, v2);
	} else
	  if ((X_test[i] == -1) && (X_test[j] == -1)) {
	    cases = 3;
	    v1 = MIN(a[i], C * s[j] - a[j]);
	    v2 = MAX(a[i] - C * s[i], - a[j]);
	    if (v2 > v1)
	      nu = 0;
	    else
	      nu = (v1+v2)/2;
	    //nu = MAX(v2, v1);
	    // printf("v1 = %lf, v2 = %lf\n", v1, v2);
	  }
  } else {
    cases = 4;
    //    printf("NO NEED TO CLIP nu = %lf\n", onu);
    return onu;
  }  
  //  printf("CLIPPED nu = %lf\n", nu);
  return nu;
}

double UpdatedDOF(int p, int q, double pap, double paq, double nap, double naq) {

  double sum1 = 0;
  double sum2 = 0;
  double sum3 = 0;
  double sum4 = 0;

  for(int j = 0; j < trsize; j++)
    if ((j != p) && (j != q)) {
      sum1 += (pap * X_test[p] * a[j] * X_test[j] * K[j][p]);
      sum2 += (paq * X_test[q] * a[j] * X_test[j] * K[j][q]);
      sum3 += (nap * X_test[p] * a[j] * X_test[j] * K[j][p]);
      sum4 += (naq * X_test[q] * a[j] * X_test[j] * K[j][q]);
    }

  double diff = - pap - paq + nap + naq + sum1 + sum2 - sum3 - sum4 +
    pap * X_test[p] * paq * X_test[q] * K[p][q] - nap * X_test[p] * naq * X_test[q] * K[p][q] +
    0.5 * (pap * pap * K[p][p] + paq * paq * K[q][q] - nap * nap * K[p][p] - naq * naq * K[q][q]);

  //  printf("dof = %lf, diff = %lf\n", dof, diff);
  return dof + diff;

}

// int SP(int i) {
//   double bdof = dof;
//   int bj = 0;

//   int l = trsize;
  
//   double sum1 = 0;
//   for(int k = 0; k < l; k++)
//     sum1 += a[k] * X_test[k] * K[k][i];
      
//   for(int j = 0; j < l; j++) {
//     if (i != j) {

//       double sum2 = 0;
      
//       for(int k = 0; k < l; k++)
// 	sum2 += a[k] * X_test[k] * K[k][j];
    
//       double nu = (X_test[j] - X_test[i] + sum1 - sum2) / (K[i][i] - 2 * K[i][j] + K[j][j]);

//       double pai = a[i];
//       double paj = a[j];
    
//       double nai = a[i] + nu * X_test[i];
//       double naj = a[j] - nu * X_test[j];

//       nu = clip(nu, nai, naj, i, j);

//       //      printf("clipped nu= %lf\n", nu);

//       //      printf("i = %d, j = %d, pai = %lf, paj = %lf, a[i] = %lf, a[j] = %lf, orig_dof = %lf\n", i, j, pai, paj, a[i], a[j], orig_dof);
//       a[i] = a[i] + nu * X_test[i];
//       a[j] = a[j] - nu * X_test[j];

//       double udof = UpdatedDOF(i, j, pai, paj, a[i], a[j]);
      
//       //      if (fabs(udof - orig_dof) > 1E-4)
//       //printf("i = %d, j = %d, pai = %lf, paj = %lf, a[i] = %lf, a[j] = %lf, udof = %lf, orig_dof = %lf\n", i, j, pai, paj, a[i], a[j], udof, orig_dof);
//       if (udof > bdof) {
// 	bdof = udof;
// 	bj = j;
//       }
      
//       a[i] = pai;
//       a[j] = paj;

//     }
//   }
//   return bj;
// }

double OptimizeTrainingAndEvalTest() { 
  double bdof =-1E20;
  double pdof = 0;
  double pof = 1E20;
  double minEps = 1E-2;
  double epsilon = 1E10;

  int nIter = 0;
  int nTotIter = 0;
  int nDecr = 0;
  int nTotDecr = 0;
  
  int l = trsize;

  for(int i = 0; i < l; i++) {
    ba[i] = a[i] = 0;
    xi[i] = 0;
    f[i] = 0;
  }
  
  bpb = pb = b = bb = 0;

  dof = 0;
  
  do {
    for (int i = 0; i < l; i++) {
      for (int j = i + 1; j < l; j++) 
	if (i != j) {
	
	  double sum1 = 0;
	  double sum2 = 0;
      
	  for(int k = 0; k < l; k++) {
	    sum1 += a[k] * X_test[k] * K[k][i];
	    sum2 += a[k] * X_test[k] * K[k][j];
	  }
      
	  double nu = (X_test[j] - sum2 - X_test[i] + sum1) / (K[i][i] - 2 * K[i][j] + K[j][j]);

	  //printf("nu = %lf\n", nu);
	  
	  double tol = 1E-7;
	  double nai = a[i] + nu * X_test[i];
	  double naj = a[j] - nu * X_test[j];

	  //

	  //	  printf("nu BEFORE clipping = %lf\n", nu);
	  //	  printf("NEW vars before clip: nai = %lf, naj = %lf\n", nai, naj);
      
	  nu = clip(nu, nai, naj, i, j, nIter);
	  // printf("nu AFTER clipping = %lf\n", nu);
	  if (nu != 0.0) {
	    //	    printf("CASE: %d, nu after clipping = %lf\n", cases, nu);
      
	    double pai = a[i];
	    double paj = a[j];

	    // printf("BEFORE CHANGE : %d, %d, pai = %lf, paj = %lf\n", i, j, pai, paj);
      
	    a[i] += nu * X_test[i];  // TRUE VAR CHANGE
	    a[j] -= nu * X_test[j];

	    //	    printf("AFTER CHANGE : %d, %d, ai = %lf, aj = %lf\n", i, j, a[i], a[j]);
	         
	    if ((a[i] < -tol) || (a[j] < -tol) || (a[i] > C * s[i] + tol) || (a[j] > C * s[j] + tol)) {
	      printf("nIter = %d, CASE :%d, ERRRORRRR CLIP ERROR NEW vars: i = %d, j = %d\n", nIter, cases, i, j);
	      printf("CASE :%d, RRRORRRR CLIP ERROR: %d %d, csi = %lf, csj = %lf, nu = %lf, nai = %lf, naj = %lf\n", cases, i, j, C*s[i], C*s[j], nu, nai, naj);
	      printf("CASE :%d, ERRRORRRR CLIP ERROR NEW vars: ai = %lf, aj = %lf\n", cases, a[i], a[j]);
	      printf("CASE :%d, ERRRORRRR CLIP ERROR OLD vars: pai = %lf, paj = %lf\n", cases, pai, paj);
	      exit(0);
	    }
	    //      printf("a[%d] = %lf\n", i, a[i]);

	    // Check constraint on b
	    // double sum = 0;
	    // for(int j = 0; j < l; j++) {
	    // 	sum += X_test[j] * a[j];
	    // }
	    // if (fabs(sum) > 1E-6 )
	    // 	printf("b constraint violation= %lf\n", sum);

	    //

	    Update_f(i, j, pai, paj, a[i], a[j], pb, b);

	    pb = b;

	    dof = UpdatedDOF(i, j, pai, paj, a[i], a[j]);

	    int n = 0;
	    double sumb = 0;
	    for(int p = 0; p < l; p++) {
	      //	      printf("s[p] = %lf\n", s[p]);
	      if ( (a[p] > 0) && (a[p] < C * s[p]) ) {
		sumb += (X_test[p] - f[p] + b);
		n++;
	      }
	    }
	    if (n > 0)
	      b = sumb /(double) n;
	    else
	      b = 0;
	    
	    if (dof > bdof) {
	      bdof = dof;
	      // printf("IMPROVED bdof = %lf, pof = %lf, eps = %lf\n", bdof, pof, epsilon);
	      for(int p = 0; p < l; p++) {
		ba[p] = a[p];
		bf[p] = f[p];
	      }
	      
	      bpb = pb;
	      bb = b;
	      pdof = dof;
	      nDecr = 0;
	    } else
	      if (dof <= bdof) {
		nDecr++;

		if (nDecr == 10) {
		  //		  printf("CHANGED CONTEST!!\n");
		  for(int p = 0; p < l; p++) {
		    a[p] = ba[p];
		    f[p] = bf[p];
		  }

		  pb = bpb ;
		  
		  b = bb  ;

		  nDecr = 0;

		  dof = pdof;
		}
	      }
	    nIter++;   // Update iteration, nu != 0
	  }
	  //	  if (nTotIter % 10 == 0)
	  //  printf("nIter = %d, dof = %.15f, b = %.15f\n", nIter, dof, b);
	  nTotIter++;
	  fflush(stdout);
	  /*	  if ((nIter > 10 * l) && (dof<-1E7)) {
	    nTotIter = 50 * l * l;
	    i = j = l;
	    }*/
	}
    }
    //    if (nTotDecr >= l * l) {
    // patience++;
      //      printf("Incremented patience = %d, dual = %lf\n", patience, dof);
    // }
  } while(nTotIter < 50 * l * l);
  
  //  printf("nIter = %d, nTotIter = %d\n", nIter, nTotIter);
  // printf("bdof = %lf\n", bdof);

  // double sumba = 0 ;
  // for(int i = 0; i < l; i++) {
  //   sumba += ba[i];
  //   printf("ba = %lf, bb = %lf\n", ba[i], bb);
  // }
  // printf("sumba = %lf, bb = %lf\n", sumba, bb);
  // exit(0);  
  // totBdof += bdof;
  // nDuals++;
  
  int lte = tesize;

  int tp = 0;
  int fp = 0;
  int fn = 0;
  int tn = 0;
  double precision;
  double recall;

  for(int i = 0; i < lte; i++) {
    // Determine F1

    f[i] = bb;
    
    for(int j = 0; j < l; j++)
      f[i] += ba[j] * X_test[j] * exp( -gammaK * (trDot(j,j) - 2 * trteDot(j, i) + teteDot(i, i))) ;

    //    printf("f = %lf, y = %lf\n", f[i], y_test[i]);
    
    if ((y_test[i] == 1.0) && (f[i] >0))
      tp++;
    if ((y_test[i] == -1.0) && (f[i] < 0))
      tn++;
    if ((y_test[i] == 1.0) && (f[i] < 0))
      fn++;
    if ((y_test[i] == -1.0) && (f[i] > 0))
      fp++;
      
  }

  if (tp + fp > 0)
    precision = tp / (double)(tp+fp);
  else
    precision = 0;

  if (tp + fn > 0)
    recall = tp / (double)(tp+fn);
  else
    recall = 0;
  
  //  printf("tp = %d, tn = %d, fn = %d, fn = %d, precision = %lf, recall=%lf\n", tp, tn, fp, fn, precision, recall);
  
  double F1;
  if (precision + recall > 0)
    F1 = 2 * (precision * recall) / (precision+recall);
  else
    F1 = 0;
        
  return F1;
}

int main(int argc, char **argv) {

  strcpy(dsname, argv[1]);
  nFeat = atof(argv[2]);
  
  // 5-fold cross-validation

  double meanF1 = 0;
  
  for(int fold = 0; fold < 5; fold++) {
    
    LoadAndPrepareData(fold);

    double BestF1 = -10;
    double BestGammaK = 0;
    double BestGammaS = 0;
    double BestC = 0;
    double BestFactor = 0;

    totBdof = 0;
    nDuals = 0;

    for(int k = 0; k < 6; k++) {
      gammaK = pow(10, 1 - k);
      //      gammaK = 0.1;
      DetermineK();
      for(int s = 0; s < 6; s++) {
	gammaS = pow(10, 4 - s);
	//gammaS = 1000;
	tot_s = 0;
	Determine_s();
	for(int c = 0; c < 7; c++) {
	  C = pow(10, 5 - c);
	  //C = 100;
	  printf("FOLD: %d, Hyper-parameters: gammaK = %lf, gammaS = %lf, C = %lf\n", fold, gammaK, gammaS, C);
	  
	  double F1 = OptimizeTrainingAndEvalTest();
	 
	  //	  printf("F1 = %lf, BestF1 = %lf\n", F1, BestF1);
            
	  if (F1 > BestF1) {
	    BestF1 = F1;
	    BestGammaK = gammaK;
	    BestGammaS = gammaS;
	    BestC = C;
	    BestFactor = tot_s / (double) trsize;
	    printf("BestF1 = %lf, BestGammaK = %lf, BestGammaS = %lf, BestC = %lf, Best w factor = %lf\n",
		   BestF1, BestGammaK, BestGammaS, BestC, BestFactor);
	  }
	  //	  exit(0);
	}
      }
    }
    //    printf("MEAN DUAL VALUE = %lf\n", totBdof/(double)nDuals);
    printf("FINISHED FOLD: %d, BestF1 = %lf, BestGammaK = %lf, BestGammaS = %lf, BestC = %lf, Best s factor = %lf\n",
	   fold, BestF1, BestGammaK, BestGammaS, BestC, BestFactor);
    fflush(stdout);
    meanF1 += BestF1;
  }

  printf("5-Fold MeanF1 = %lf\n", meanF1 / 5.0);
  fflush(stdout)  ;
  return 0;
}

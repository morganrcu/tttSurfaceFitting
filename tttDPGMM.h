#ifndef tttDPGMM_H_
#define tttDPGMM_H_

#include <vnl/vnl_matrix.h>
#include <vnl/vnl_vector.h>
#include <vnl/vnl_math.h>
#include <vnl/vnl_random.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <math.h>
#include <boost/math/special_functions/digamma.hpp>

template<class T> T digamma( const T & x) {
	T result=x;
	for(int k=0;k<x.size();k++ ){
		result[k]=boost::math::digamma(x[k]);
	}
	return result;
}

template<> double digamma(const double & x){
	return boost::math::digamma(x);
}
template<class T> T gammaln(const T & x) {
	T result=x;
	for(int k=0;k<x.size();k++ ){
		result[k]=std::lgamma(x[k]);
	}
	return result;
}
template<> double gammaln(const double & x){
	return std::lgamma(x);
}


class DPGMM {

	typedef vnl_matrix<double> MatrixType;
	typedef vnl_vector<double> VectorType;
public:
	enum CovarianceType {
		DPGMM_COVARIANCE_DIAG, DPGMM_COVARIANCE_FULL, DPGMM_COVARIANCE_SPHERICAL,DPGMM_COVARIANCE_TIED
	};

	DPGMM(int numberOfComponents = 1, CovarianceType covarianceType =
			DPGMM_COVARIANCE_DIAG, double alpha = 1.0,unsigned maxIterations=20) {
		m_MaxIterations=maxIterations;
		m_NumberOfComponents=numberOfComponents;
		m_CovarianceType = covarianceType;
		m_Alpha=alpha;
		m_MaxIterations=maxIterations;
		m_NumberOfKMeansIterations=10;
		m_A0=1.0;
		m_B0=1.0;


	}

	MatrixType getMeans(){
		return this->m_Means;
	}

public:

	VectorType sumRows(const MatrixType & matrix){
		VectorType result(matrix.cols());
		result.fill(0.0);
		for(int c=0;c<matrix.cols();c++){
			result(c)=matrix.get_column(c).sum();
		}
		return result;
	}

	MatrixType boundStateLogLikelihood(const MatrixType & X,const VectorType & initialBound,MatrixType & means, const std::vector<MatrixType> & precs ){
		unsigned nFeatures = means.rows();
		unsigned nSamples = X.rows();

		MatrixType bound(X.rows(),this->m_NumberOfComponents);
		for(int n=0;n<X.rows();n++){
			bound.set_row(n,initialBound);
		}


		switch(m_CovarianceType){
		case DPGMM_COVARIANCE_DIAG:
		case DPGMM_COVARIANCE_SPHERICAL:
		{
			for(int k=0;k<this->m_NumberOfComponents;k++){
				//std::cout <<"Prec" << k << "\t"<< precs[k] <<std::endl;
				//std::cout <<"Mean" << k << "\t"<< means.get_column(k) <<std::endl;

				for(int n=0;n<nSamples;n++){
					auto d = X.get_row(n) - means.get_column(k);

					bound(n,k)-=0.5*dot_product(d,d)*precs[k](0,0);
				}
			}
			break;
		}
		case DPGMM_COVARIANCE_TIED:
		{
			break;
		}

		case DPGMM_COVARIANCE_FULL:
		{
			break;
		}
		}
		return bound;
	}

	void updateMeans(const MatrixType & X, const MatrixType & z){
		unsigned nFeatures = X.cols();

		this->m_Means.set_size(nFeatures,this->m_NumberOfComponents);
		this->m_Means.fill(0);

		for(int k=0;k<m_NumberOfComponents;k++){

			switch(m_CovarianceType){
			case DPGMM_COVARIANCE_DIAG:
			case DPGMM_COVARIANCE_SPHERICAL:
			{
				VectorType num(nFeatures);
				num.fill(0);
				double den=0;
				for(int n=0;n<X.rows();n++){
					num+= z(n,k)*X.get_row(n);
					den+= z(n,k);
				}
				den=this->m_Precisions[k](0,0)*den+1;


				num=num*this->m_Precisions[k](0,0);

				VectorType mean= num/den;
				this->m_Means.set_column(k,mean);

				break;
			}
			}
		}

	}
	void updateConcentration(const MatrixType & Z){

		//VectorType sz = this->sumRows(z);

		VectorType sz(this->m_NumberOfComponents);
		sz.fill(0);
		for(int n=0;n<Z.rows();n++){
			sz+=Z.get_row(n);
		}

		this->m_Gamma1= sz+1;
		this->m_Gamma2.set_size(this->m_NumberOfComponents);
		this->m_Gamma2.fill(0);

		for(int k=m_NumberOfComponents-2;k>=0;k--){
			this->m_Gamma2[k]=this->m_Gamma2[k+1]+sz[k];
		}
		this->m_Gamma2+= this->m_Alpha;

	}

	void updatePrecisions(const MatrixType & X,const MatrixType & Z){
		unsigned numFeatures = X.cols();

		this->m_Scale.set_size(m_NumberOfComponents);
		this->m_Bounded_Precisions.set_size(m_NumberOfComponents);

		switch(m_CovarianceType){
		case DPGMM_COVARIANCE_SPHERICAL:
		{
			this->m_DOF =m_A0 + 0.5*numFeatures*this->sumRows(Z);

			for(int k=0;k<m_NumberOfComponents;k++){

				this->m_Scale[k]=m_B0;

				VectorType sumSqDiff(X.rows());
				sumSqDiff.fill(0);

				for(int n=0;n<X.rows();n++){
					auto diff = (X.get_row(n)-m_Means.get_column(k));

					sumSqDiff(n)=diff.apply(vnl_math_sqr).sum();
				}
				this->m_Scale[k]+=0.5*dot_product(Z.get_column(k),(sumSqDiff+numFeatures));

				this->m_Bounded_Precisions[k]=(0.5*numFeatures*(digamma(this->m_DOF[k])-std::log(this->m_Scale[k])));
			}

			for(int k=0;k<m_NumberOfComponents;k++){
				this->m_Precisions[k](0,0)=this->m_DOF[k]/this->m_Scale[k];
				std::cout << this->m_DOF[k]<< "/"<< this->m_Scale[k] << "=" << this->m_Precisions[k](0,0) << std::endl;

			}
			//  self.dof_ = 0.5 * n_features * np.sum(z, axis=0)
            //for k in range(self.n_components):
			//  # could be more memory efficient ?
			//   sq_diff = np.sum((X - self.means_[k]) ** 2, axis=1)
			//   self.scale_[k] = 1.
			//    self.scale_[k] += 0.5 * np.sum(z.T[k] * (sq_diff + n_features))
			//   self.bound_prec_[k] = (
			//        0.5 * n_features * (
			//            digamma(self.dof_[k]) - np.log(self.scale_[k])))
			//self.precs_ = np.tile(self.dof_ / self.scale_, [n_features, 1]).T
		}
		}

	}
	void doMStep(const MatrixType & X, const MatrixType & z){
		this->updateConcentration(z);
		this->updateMeans(X,z);
		this->updatePrecisions(X,z);

	}

	MatrixType logNormalize(const MatrixType & matrix){
		MatrixType result=matrix;

		for(int n=0;n<matrix.rows();n++){
			result.set_row(n,result.get_row(n)-result.get_row(n).max_value());
		}

		double max = result.max_value();
		double sum = 0;
		for(int r=0;r<result.rows();r++){
			for(int c=0;c<result.cols();c++){
				sum+=std::exp(result(r,c)-max);
			}
		}

		double out =std::log(sum) +max;

		for(int r=0;r<result.rows();r++){
			for(int c=0;c<result.cols();c++){
				result(r,c)=std::exp(result(r,c)-out) +std::numeric_limits<double>::epsilon();
			}
			result.set_row(r,result.get_row(r)/result.get_row(r).sum());
		}
		return result;
	}

	void scoreSamples(const MatrixType & X,VectorType & logprob, MatrixType & z,VectorType & bound){
		//TODO check is fitted
		unsigned nSamples = X.rows();
		z.set_size(X.rows(),m_NumberOfComponents);
		z.fill(0.0);
		VectorType sd = digamma(this->m_Gamma1+this->m_Gamma2);

		auto dgamma1= digamma(this->m_Gamma1) -sd;
		VectorType dgamma2(m_NumberOfComponents);
		dgamma2[0]=digamma(this->m_Gamma2[0])-digamma(this->m_Gamma1[0]+this->m_Gamma2[0]);
		for(int k=1;k<m_NumberOfComponents;k++){
			dgamma2[k]=dgamma2[k-1]+digamma(this->m_Gamma2[k-1]);
			dgamma2[k]-=sd[k-1];
		}
		auto dgamma = dgamma1+dgamma2;
		dgamma1.clear();
		dgamma2.clear();
		sd.clear();

		MatrixType p=this->boundStateLogLikelihood(X,this->m_InitialBound+this->m_Bounded_Precisions,this->m_Means,this->m_Precisions);

		for(int n=0;n<nSamples;n++){
			z.set_row(n,p.get_row(n)+dgamma);
		}
		z=logNormalize(z);
		//std::cout << "Z" << std::endl;
		//std::cout << z << std::endl;
		bound.set_size(m_NumberOfComponents);
		bound.fill(0.0);
		auto zp = element_product(z,p);
		for(int n=0;n<nSamples;n++){
			bound+=zp.get_row(n);
		}

	}
	void initializeGamma(const VectorType & w, const MatrixType & Z){
		m_Gamma0.set_size(m_NumberOfComponents);
		m_Gamma0.fill(m_Alpha);
#if 0
		m_Gamma1.set_size(m_NumberOfComponents);
		m_Gamma1.fill(m_Alpha);

		m_Gamma2.set_size(m_NumberOfComponents);
		m_Gamma2.fill(m_Alpha);

#endif
#if 1
		VectorType sz(this->m_NumberOfComponents);
		sz.fill(0);
		for(int n=0;n<w.size();n++){
			sz+=w[n]*Z.get_row(n);
		}

		this->m_Gamma1= sz+1;
		this->m_Gamma2.set_size(this->m_NumberOfComponents);
		this->m_Gamma2.fill(0);

		for(int k=m_NumberOfComponents-2;k>=0;k--){
			this->m_Gamma2[k]=this->m_Gamma2[k+1]+sz[k];
		}
		this->m_Gamma2+= this->m_Alpha;
#endif
	}

	void initZKMeans(const MatrixType & X, const VectorType & w, MatrixType & Z){

		unsigned numSamples = X.rows();
		unsigned numFeatures = X.cols();

		Z.set_size(numSamples,this->m_NumberOfComponents);


		vnl_vector<unsigned> assignments(numSamples);

		for(int n=0;n<numSamples;n++){
			assignments[n]=m_RNG(this->m_NumberOfComponents);
		}

		MatrixType centers(numFeatures,m_NumberOfComponents);

		for(int it=0;it<m_NumberOfKMeansIterations;it++){

			centers.fill(0);
			//Compute centroids
			vnl_vector<double> counts(m_NumberOfComponents);
			counts.fill(0);

			for(int n=0;n<numSamples;n++){
				unsigned assignment = assignments[n];
				centers.set_column(assignment,centers.get_column(assignment)+w[n]*X.get_row(n));
				counts[assignment]+=w[n];
			}
			//std::cout << counts << std::endl;

			for(int k=0;k<m_NumberOfComponents;k++){
				if(counts[k]>0){
					centers.set_column(k,centers.get_column(k)/counts[k]);
				}else{
					for(int r=0;r<centers.rows();r++){
						centers(r,k)=m_RNG.normal();
					}
				}
			}

			std::cout << centers << std::endl;

			//Compute assignments
			Z.fill(0.0);
			for(int n=0;n<numSamples;n++){
				VectorType distances(this->m_NumberOfComponents);
				distances.fill(0);

				for(int k=0;k<m_NumberOfComponents;k++){
					distances[k]=(X.get_row(n)-centers.get_column(k)).magnitude();
				}
				assignments[n]=distances.arg_min();
				Z(n,distances.arg_min())=1.0;
			}
		}
	}

	void initializeMeans(const MatrixType & X,const VectorType & w,const MatrixType & Z){
		int numSamples = X.rows();
		int numFeatures= X.cols();
		this->m_Means.set_size(numFeatures,this->m_NumberOfComponents);
		VectorType counts(this->m_NumberOfComponents);
		counts.fill(0.0);
		this->m_Means.fill(0);

		for(int n=0;n<numSamples;n++){
			for(int k=0;k<this->m_NumberOfComponents;k++){
				this->m_Means.set_column(k,this->m_Means.get_column(k)+Z(n,k)*w[n]*X.get_row(n));
				counts(k)+=Z(n,k)*w[n];
			}
		}


		for(int k=0;k<m_NumberOfComponents;k++){
			if(counts[k]>0){
				this->m_Means.set_column(k,this->m_Means.get_column(k)/counts[k]);
			}else{
				for(int r=0;r<m_Means.rows();r++){
					this->m_Means(r,k)=m_RNG.normal();
				}
			}
		}
		std::cout << "Init m_Means" << std::endl;
		std::cout << this->m_Means << std::endl;

	}

	void initializePrecisions(const MatrixType & X,const VectorType & w,const MatrixType & Z){
		unsigned numFeatures =X.cols();

		switch(m_CovarianceType){
		case DPGMM_COVARIANCE_SPHERICAL:
			this->m_DOF.set_size(this->m_NumberOfComponents);
			this->m_DOF.fill(this->m_A0);

			this->m_Scale.set_size(this->m_NumberOfComponents);
			this->m_Scale.fill(this->m_B0);

			this->m_Precisions.resize(this->m_NumberOfComponents);

			std::for_each(m_Precisions.begin(),m_Precisions.end(),[](MatrixType & matrix){
				matrix.set_size(1,1);
				//matrix.fill(1.0);
				matrix.fill(0);
			});

			for(int n=0;n<X.rows();n++){
				for(int k=0;k<this->m_NumberOfComponents;k++){
					auto diff = X.get_row(n) - this->m_Means.get_column(k);
					this->m_Scale[k]+=Z(n,k)*w[n]*diff.apply(vnl_math_sqr).sum();
					this->m_DOF[k]+=Z(n,k)*w[n];
				}
			}
			for(int k=0;k<this->m_NumberOfComponents;k++){
				this->m_Precisions[k](0,0)=this->m_DOF[k]/this->m_Scale[k];
				std::cout << this->m_DOF[k]<< "/"<< this->m_Scale[k] << "=" << this->m_Precisions[k](0,0) << "\t";
			}
			std::cout << std::endl;
			auto logScale = m_Scale.apply(std::log);
			auto digammaDOF = digamma(m_DOF);
			auto digmmaDOF_logScale= digammaDOF-logScale;
			this->m_Bounded_Precisions=0.5*numFeatures*digmmaDOF_logScale;
#if 0
			this->m_DOF.set_size(this->m_NumberOfComponents);
			this->m_DOF.fill(1.0);
			this->m_Scale.set_size(this->m_NumberOfComponents);
			this->m_Scale.fill(1.0);
			this->m_Precisions.resize(this->m_NumberOfComponents);

			std::for_each(m_Precisions.begin(),m_Precisions.end(),[](MatrixType & matrix){
				matrix.set_size(1,1);
				//matrix.fill(1.0);
				matrix.fill(100);
			});

			auto logScale = m_Scale.apply(std::log);
			auto digammaDOF = digamma(m_DOF);
			auto digmmaDOF_logScale= digammaDOF-logScale;
			this->m_Bounded_Precisions=0.5*numFeatures*digmmaDOF_logScale;

			//self.dof_ = np.ones(self.n_components)
		    //self.scale_ = np.ones(self.n_components)
		    //self.precs_ = np.ones((self.n_components, n_features))
		    //self.bound_prec_ = 0.5 * n_features * ( digamma(self.dof_) - np.log(self.scale_))
#endif
		}
	}
	void initializeWeights(){
		this->m_Weights.set_size(this->m_NumberOfComponents);
		this->m_Weights.fill(1.0/this->m_NumberOfComponents);
	}


	void init( const MatrixType & X,const VectorType & w){

		unsigned numFeatures =X.cols();

		m_InitialBound=-0.5*numFeatures*std::log(2*M_PI)- std::log(2*M_PI*M_E);

		MatrixType Z;
		this->initZKMeans(X,w,Z);

		this->initializeGamma(w,Z);
		this->initializeMeans(X,w,Z);
		this->initializeWeights();
		this->initializePrecisions(X,w,Z);
	}

	double boundConcentration(){
		double logprior = gammaln(this->m_Alpha)*this->m_NumberOfComponents;

		logprior+=((m_Alpha-1)*(digamma(this->m_Gamma2)-digamma(this->m_Gamma1+this->m_Gamma2))).sum();
		logprior+=(-gammaln(this->m_Gamma1+this->m_Gamma2)).sum();
		logprior+=(gammaln(this->m_Gamma1)+gammaln(this->m_Gamma2)).sum();
		logprior-=element_product(this->m_Gamma1-1,(digamma(this->m_Gamma1)-digamma(this->m_Gamma1+this->m_Gamma2))).sum();
		logprior-=element_product(this->m_Gamma2-1,(digamma(this->m_Gamma2)-digamma(this->m_Gamma2-digamma(this->m_Gamma1+this->m_Gamma2)))).sum();
		return logprior;
	}

	double boundMeans(){
		double logprior=0;

		logprior -= 0.5*pow(this->m_Means.frobenius_norm(),2);
		logprior -= 0.5*this->m_Means.rows()*this->m_NumberOfComponents;
		return logprior;

	}

	double boundPrecisions(){
		double logprior=0;
		switch(this->m_CovarianceType){
		case DPGMM_COVARIANCE_SPHERICAL:
		{
			logprior +=gammaln(this->m_DOF).sum();
			logprior -=element_product(this->m_DOF-1,digamma(this->m_DOF.apply([](double v){return v<0.5?0.5:v;}))).sum();
			logprior +=(this->m_Scale.apply(std::log) + this->m_DOF).sum();

			for(int k=0;k<this->m_NumberOfComponents;k++){
				logprior+=this->m_Precisions[k](0,0);
			}
			break;
		}
		}
		return logprior;
	}

	double boundProportions(const MatrixType & z){
		auto dg12= digamma(this->m_Gamma1+this->m_Gamma2);
		auto dg1 = digamma(this->m_Gamma1) -dg12;
		auto dg2 = digamma(this->m_Gamma2) -dg12;

		vnl_vector<double> cz(z.rows());
		cz.fill(0);
		//TODO

	}
	void logPrior(const MatrixType & z){
		this->boundConcentration();
		this->boundMeans();
		this->boundPrecisions();
		this->boundProportions(z);

	}

public:




	void train(MatrixType & X,VectorType & w) {
		this->init(X,w);

		for(unsigned it=0;it< m_MaxIterations;++it){
			//Expectation step
			VectorType currentLogprob;
			MatrixType z;
			VectorType bound;
			this->scoreSamples(X,currentLogprob,z,bound);
			//std::cout << z << std::endl;
			//FIXME double currentLogLikelihood=currentLogprob.mean() +this->logPrior(z)/z.size();
			//std::cout << bound << std::endl;
			//TODO check for convergence

			for(int n=0;n<z.rows();n++){
				z.set_row(n,z.get_row(n)*w[n]);
			}
			this->doMStep(X,z);

			std::cout << "Mean:" << std::endl;
			std::cout << this->m_Means << std::endl;
			std::cout << "Gamma1" << std::endl;
			std::cout << this->m_Gamma1 << std::endl;
#if 0
			std::cout << "Precisions:" << std::endl;
			for(int k=0;k<m_NumberOfComponents;k++){
				std::cout << this->m_Precisions[k] << std::endl;
			}
#endif
#if 0
			std::cout << "Gamma:" << std::endl;
			std::cout << this->m_Gamma1 << std::endl;
			std::cout << this->m_Gamma2 << std::endl;
#endif
		}
	}

protected:

public:
	unsigned	m_MaxIterations;
	unsigned	m_NumberOfKMeansIterations;
	unsigned	m_NumberOfComponents;

	CovarianceType m_CovarianceType;
	VectorType m_Gamma0;
	VectorType m_Gamma1;
	VectorType m_Gamma2;

	MatrixType m_Means;



	vnl_random m_RNG;

	VectorType m_DOF;
	VectorType m_Scale;
	std::vector<MatrixType> m_Precisions;
	VectorType m_Bounded_Precisions;

	VectorType m_Weights;
	double m_Alpha;
	double m_A0;
	double m_B0;

	double m_InitialBound;
};

#endif

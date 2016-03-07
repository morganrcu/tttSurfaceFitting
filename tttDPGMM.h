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
			DPGMM_COVARIANCE_DIAG, double alpha = 1.0,unsigned maxIterations=100) {
		m_MaxIterations=maxIterations;
		m_NumberOfComponents=numberOfComponents;
		m_CovarianceType = covarianceType;
		m_Alpha=alpha;
		m_MaxIterations=maxIterations;
		m_NumberOfKMeansIterations=10;


	}

	MatrixType getMeans(){
		return this->m_Means;
	}

protected:

	VectorType sumRows(const MatrixType & matrix){
		VectorType result(matrix.cols());

		for(int c=0;c<matrix.cols();c++){
			result[c]=matrix.get_column(c).sum();
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
				for(int n=0;n<nSamples;n++){
					auto d = X.get_row(n) - means.get_column(k);
					bound(n,k)=-0.5*dot_product(d,(precs[k]*d));
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

	void updateConcentration(){

	}

	void updateMeans(const MatrixType & X, const VectorType & w,const MatrixType & z){
		unsigned nFeatures = X.cols();

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
					num+= z(n,k)*w[n]*X.get_row(n);
					den+= z(n,k)*w[n];
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
	void updateConcentration(const MatrixType & z){

		VectorType sz = this->sumRows(z);
		this->m_Gamma1= sz+1;
		this->m_Gamma2.fill(0);

		for(int k=m_NumberOfComponents-2;k>=0;k--){
			this->m_Gamma2[k]=this->m_Gamma2[k+1]+sz[k];
		}
		this->m_Gamma2+= this->m_Alpha;

	}

	void updatePrecisions(const MatrixType & X, const VectorType & w,const MatrixType & z){

		unsigned numFeatures = X.cols();

		switch(m_CovarianceType){
		case DPGMM_COVARIANCE_SPHERICAL:
		{
			this->m_DOF = 0.5*numFeatures*this->sumRows(z);

			for(int k=0;k<m_NumberOfComponents;k++){
				this->m_Scale[k]=1;
				for(int n=0;n<X.rows();n++){
					auto sq_diff = (X.get_row(n)-m_Means.get_column(k));
					sq_diff=sq_diff.apply(vnl_math_sqr);
					this->m_Scale[k]+=0.5*z(n,k)*w[n]*(sq_diff+numFeatures).sum();

				}
				this->m_Bounded_Precisions[k]=(0.5*numFeatures*(digamma(this->m_DOF[k])-std::log(this->m_Scale[k])));
			}
			for(int k=0;k<m_NumberOfComponents;k++){
				this->m_Precisions[k](0,0)=this->m_DOF[k]/this->m_Scale[k];
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
	void doMStep(const MatrixType & X, const VectorType & w, const MatrixType & z){
		this->updateConcentration(z);
		this->updateMeans(X,w,z);
		//this->updatePrecisions(X,w,z);

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

		bound.set_size(m_NumberOfComponents);
		bound.fill(0.0);
		auto zp = element_product(z,p);
		for(int n=0;n<nSamples;n++){
			bound+=zp.get_row(n);
		}
		std::cout << "Bound\t" << bound << std::endl;
#if 0
		double sd=digamma();
		double dgamma1=di;
		double dgamma2=di;
		double dgamma2[0]
#endif
		//this->boundStateLogLikelihood(X,this->m)


	}
	void initializeGamma(){
		m_Gamma0.set_size(m_NumberOfComponents);
		m_Gamma0.fill(m_Alpha);

		m_Gamma1.set_size(m_NumberOfComponents);
		m_Gamma1.fill(m_Alpha);

		m_Gamma2.set_size(m_NumberOfComponents);
		m_Gamma2.fill(m_Alpha);
	}
	void initializeMeans(const MatrixType & X,const VectorType & w){

		unsigned numSamples = X.rows();
		unsigned numFeatures = X.cols();

		vnl_vector<unsigned> assignments(numSamples);

		vnl_random rng;
		for(int n=0;n<numSamples;n++){
			assignments[n]=rng(this->m_NumberOfComponents);
		}

		this->m_Means.set_size(numFeatures,m_NumberOfComponents);
		this->m_Means.fill(0);
		vnl_vector<double> counts(m_NumberOfComponents);
		counts.fill(0);
		for(int it=0;it<m_NumberOfKMeansIterations;it++){

			//Compute centroids
			for(int n=0;n<numSamples;n++){
				unsigned assignment = assignments[n];
				this->m_Means.set_column(assignment,this->m_Means.get_column(assignment)+w[n]*X.get_row(assignment));
				counts[assignment]+=w[n];
			}

			for(int k=0;k<m_NumberOfComponents;k++){
				this->m_Means.set_column(k,this->m_Means.get_column(k)/counts[k]);
			}

			std::cout << m_Means << std::endl;

			//Compute assignments
			for(int n=0;n<numSamples;n++){
				VectorType distances(this->m_NumberOfComponents);
				distances.fill(0);

				for(int k=0;k<m_NumberOfComponents;k++){
					distances[k]=(X.get_row(n)-this->m_Means.get_column(k)).magnitude();
				}
				assignments[n]=distances.arg_min();
			}

		}

	}
	void initializePrecisions(const MatrixType & X){
		unsigned numFeatures =X.cols();

		switch(m_CovarianceType){
		case DPGMM_COVARIANCE_SPHERICAL:

			this->m_DOF.set_size(this->m_NumberOfComponents);
			this->m_DOF.fill(1.0);
			this->m_Scale.set_size(this->m_NumberOfComponents);
			this->m_Scale.fill(1.0);
			this->m_Precisions.resize(this->m_NumberOfComponents);

			std::for_each(m_Precisions.begin(),m_Precisions.end(),[](MatrixType & matrix){
				matrix.set_size(1,1);
				matrix.fill(1.0);
			});

			auto logScale = m_Scale.apply(std::log);
			auto digammaDOF = digamma(m_DOF);
			auto digmmaDOF_logScale= digammaDOF-logScale;
			this->m_Bounded_Precisions=0.5*numFeatures*digmmaDOF_logScale;

			//self.dof_ = np.ones(self.n_components)
		    //self.scale_ = np.ones(self.n_components)
		    //self.precs_ = np.ones((self.n_components, n_features))
		    //self.bound_prec_ = 0.5 * n_features * ( digamma(self.dof_) - np.log(self.scale_))
		}
	}
	void initializeWeights(){
		this->m_Weights.set_size(this->m_NumberOfComponents);
		this->m_Weights.fill(1.0/this->m_NumberOfComponents);
	}


	void init( const MatrixType & X,const VectorType & w){

		unsigned numFeatures =X.cols();

		m_InitialBound=-0.5*numFeatures*std::log(2*M_PI)- std::log(2*M_PI*M_E);

		this->initializeGamma();
		this->initializeMeans(X,w);
		this->initializeWeights();
		this->initializePrecisions(X);
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
			//FIXME double currentLogLikelihood=currentLogprob.mean() +this->logPrior(z)/z.size();
			//std::cout << bound << std::endl;
			//TODO check for convergence

			this->doMStep(X,w,z);

			std::cout << "Mean:" << std::endl;
			std::cout << this->m_Means << std::endl;
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

private:
	unsigned	m_MaxIterations;
	unsigned	m_NumberOfKMeansIterations;
	unsigned	m_NumberOfComponents;

	CovarianceType m_CovarianceType;
	VectorType m_Gamma0;
	VectorType m_Gamma1;
	VectorType m_Gamma2;

	MatrixType m_Means;


	VectorType m_DOF;
	VectorType m_Scale;
	std::vector<MatrixType> m_Precisions;
	VectorType m_Bounded_Precisions;

	VectorType m_Weights;
	double m_Alpha;

	double m_InitialBound;
};

#endif

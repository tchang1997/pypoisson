# PyPoisson

Python implementation of Poisson regression models, including:
* Univariate poisson regression
* Bivariate poisson regression 
* Bivariate poisson regression with diagonal inflation 

Double Poisson models and associated variants are ubiquitious for sports predictions, especially in soccer/football. However, implementations of standard Poisson regression methods in Python beyond the univariate case remain limited. This repository aims to bridge this gap. 

Implementations are based on:

Karlis, D., & Ntzoufras, I. (2005). Bivariate Poisson and diagonal inflated bivariate Poisson regression models in R. *Journal of statistical Software*, 14, 1-36.

I'd like to highlight the following as an inspiration for the bivariate Poisson regression implementation:

Medley, J. (2022). Bivariate Poisson Regression with Expectation-Maximisation. *Personal blog.*

## Quickstart

Run this command:
```
    python demo.py --type {univariate,double,inflated}
```
to demo the univariate, bivariate, and bivariate + diagonal-inflated Poisson regression methods on a simulated dataset. For the diagonal-inflated bivariate Poisson model, data will be cached at `data/`.

## How it works

A Poisson distribution aims to model the *rate* at which an event of interest occurs. Notated as $Poisson(\lambda)$, or $Poi(\lambda)$ as shorthand, $\lambda$ is a *parameter* that represents how often the event of interest is *expected* to occur across a given timeframe. In plain English, we can think of $\lambda$ as *events per time period.*

**Univariate case.** This is simply a generalized linear model for the Poisson distribution, for which we use the log-link function. Essentially, given some covariates $x$ and parameters $\beta$, we assume that $\log \lambda = \exp(\beta^\top x)$, and use standard maximum likelihood estimation techniques to fit $\beta$. This is included for completeness; for real use cases I recommend the `statsmodels` package.

**Bivariate case (no inflation).** This model fits a model for two Poisson distributions that may be correlated. This method has applications in sports predictions, so we will use soccer/football prediction as a running example, and specifically El Clasico. We model the number of goals that Barcelona and Real Madrid score at *Poisson random variables* (i.e., goals per game). However, the number of goals that each team scores may be dependent on one another; *i.e.*, if Real Madrid opens the scoring, Barcelona is heavily incentivized to equalize and may press harder, and vice versa. Concretely, this dependence can be modeled as a third *latent* (unobserved) variable. At a high level, we use the expectation-maximization algorithm to repeatedly impute preliminary estimates of that latent variable, and use those estimates to then predict the remaining observed parameters.

**Bivariate case (with diagonal-inflation).** This model works similarly to the above, down to using the same algorithm (expectation maximization). The key difference is *diagonal inflation.* To understand what diagonal inflation is, imagine the probabilities for two Poisson random variables laid out in a large N x N matrix. Concretely, pretend that we have a 8 x 8 matrix; each entry of this **goal matrix** represents the probability that El Clasico ends with Barcelona and Real Madrid scoring some number of goals each, from 0 to 7 (in theory, we'd have to go to an infinite # of goals such that the probabilities sum to 1, but let's say that 7 is close enough). However, when we run bivariate Poisson regression, others have found that it tends to underestimate the number of draws. Each entry of the *diagonal* of this *goal matrix* represents a draw (0-0, 1-1, 2-2, etc.) -- what if we upweighted them somehow? That's the idea behind diagonal-inflated bivariate Poisson regression: we upweight the probabilty of draws according to some distribution (*e.g.*, a geometric, another Poisson [lmao], or some discrete distribution that we specify).

For details, please refer to the code, although if there is enough interest I can be persuaded to recreate derivations/visual aids for all the models involved. This was hacked up very quickly; feel free to report any bugs and contribute to this project. 

While these methods are not original research, please link to this repository if you find it useful/take inspiration in your own work.

## Contact

`ctrenton (at) umich (dot) edu`


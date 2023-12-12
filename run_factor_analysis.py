import os
import pandas

from factor_analyzer import FactorAnalyzer

from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity

import numpy

import matplotlib.pyplot as pyplot

# ignore math scores in last column for factor analysis - this should be based just on questionnaire
dataset = pandas.read_csv("csv_files/dataset.csv").iloc[:,:-1]

print(dataset)


chi2 ,p=calculate_bartlett_sphericity(dataset)
print(chi2, p)

machine = FactorAnalyzer(n_factors=40, rotation=None)
machine.fit(dataset)
ev, v = machine.get_eigenvalues()
print(ev)


if not os.path.exists('plots'):
  os.mkdir('plots')

x = numpy.arange(0,len(ev))
pyplot.plot(x,ev, marker='o')
pyplot.ylabel("Eigenvalue")
pyplot.savefig("plots/eigenvalue_plot.png")
pyplot.show()
pyplot.close()


# consider 0 to be a factor, n = 0:5
machine = FactorAnalyzer(n_factors=6, rotation=None)
machine.fit(dataset)
output = machine.loadings_
print(output)

# 2 factors based on ev plot - explains 54.58% of the variance
machine = FactorAnalyzer(n_factors=2, rotation='varimax')
machine.fit(dataset)
factor_loadings = machine.loadings_
print(factor_loadings)

# save factor loadings data to use in determining which 20 questions should be used in assembling a team
pandas.DataFrame(factor_loadings).to_csv("csv_files/factor_loadings.csv", index=False)

dataset = dataset.values

results = numpy.dot(dataset, factor_loadings)


if not os.path.exists('csv_files'):
  os.mkdir('csv_files')
# could round results.. not knowing the context of original dataset, I prefer to use decimals for greater precision
pandas.DataFrame(results).to_csv("csv_files/traits.csv", index=False)




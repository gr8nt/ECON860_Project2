import pandas
import matplotlib.pyplot as pyplot
import numpy
import os

traits_with_math = pandas.read_csv("csv_files/traits.csv")
math = pandas.read_csv("csv_files/dataset.csv")

math_score = math["math"]
traits_with_math = traits_with_math.join(math_score)

df = pandas.DataFrame(traits_with_math)

# order columns as math score, trait1(0), trait2(1)
df = df [['math', '0', '1']]


if not os.path.exists('csv_files'):
  os.mkdir('csv_files')

df.to_csv("csv_files/traits_with_math.csv", index=False)

print(df)



# plot relationship between trait 1, trait 2, math score

if not os.path.exists('plots'):
  os.mkdir('plots')


dataset = pandas.read_csv("csv_files/traits_with_math.csv")

x = dataset['0']
y = dataset['1']
colors = dataset['math']

pyplot.scatter(x,y, s=10, c=colors)

pyplot.title("Trait 1 & 2 vs. Math Score")
pyplot.xlabel("Trait 1")
pyplot.ylabel("Trait 2")
cbar = pyplot.colorbar()
cbar.set_label("Math Score")

pyplot.savefig("plots/plot_traits_math.png")
pyplot.show()
pyplot.close()

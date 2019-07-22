# Juan Salamanca and Santiago Nunez-Corrales
# Social viscosity
#
# This file is intended to be run in RStudio

exp_all <- read.csv(file.choose())
# Choose "all_to_manova.csv"

result.manova <- manova(cbind(K, N) ~ TOL, data = exp_all)
summary(result.manova)
summary.aov(result.manova)

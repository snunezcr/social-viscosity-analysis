# Juan Salamanca and Santiago Nunez-Corrales
# Social viscosity
#
# This file is intended to be run in RStudio

exp_all <- read.csv(file.choose())
# Choose "nnn_to_manova.csv"

result.manova <- manova(cbind(K, N) ~ NNN + TOL + NNN*TOL, data = exp_all)
summary(result.manova)
summary.aov(result.manova)


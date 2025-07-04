# Depth and Temperature Fits with Gradient-Based Temperature and R^2 Output

library(doParallel)
registerDoParallel(cores = 16)
library(foreach)
library(glmnet)
library(fields)
library(nlstools)

####################################
### 1) OPEN AND FORMAT FILES
####################################
gridCells = read.csv("metadata_with_merged_depth.csv", stringsAsFactors = FALSE)
GreenlandFID = c(3774:3776,3759:3762,3737:3742,3711:3715,3681:3685,3639:3643,3584:3588,3522:3525,3452:3454,3378:3380)
AntarcFID = 3791:4163

all = read.csv("cores_with_gradient_filled.csv")
all = all[which(all$MethodCM == "direct"), ]
all$Depth = as.numeric(all$Depth)
all$cellsPer = as.numeric(all$cellsPer)

myIndices = as.matrix(read.csv("1000_indices_for_bootstrap.csv", header = FALSE))
bootstraps = nrow(myIndices)
depthsToIterate = gridCells$maxdepth * 1000

####################################
### 2) DEFINE OUTPUT STRUCTURES
####################################
temperature.error = vector(length = bootstraps)
temperature.rsq = vector(length = bootstraps)
cv.biomass = vector(length = bootstraps)
cv.byGridResult = data.frame(matrix(NA, nrow = nrow(gridCells), ncol = bootstraps))
model.coefficients = list()
bad.fids = c()

ptm <- proc.time()

for (n in 1:bootstraps) {
  if (n %% 5 == 0) cat("Bootstrap #", n, "\n")
  train_ind <- myIndices[n, ]

  all$temperature = all$mast + all$Depth * all$gradient / 1000
  print(all$temperature[1:5])

  keep = c("cellsPer", "Depth", "temperature")
  train <- all[train_ind, which(colnames(all) %in% keep)]
  test <- all[-train_ind, which(colnames(all) %in% keep)]

  train$Depth = log10(train$Depth)
  test$Dwriepth = log10(test$Depth)
  train$cellsPer = log10(train$cellsPer)
  test$cellsPer = log10(test$cellsPer)

  train = train[complete.cases(train), ]
  test = test[complete.cases(test), ]

  train.x = model.matrix(cellsPer ~ ., train)[, -1]
  train.y = train$cellsPer
  test.x = model.matrix(cellsPer ~ ., test)[, -1]
  test.y = test$cellsPer

  glmfit = glmnet(train.x, train.y)
  cv.fit = cv.glmnet(train.x, train.y)
  preds = predict(cv.fit, newx = test.x, s = "lambda.min")

  ss_res = sum((test.y - preds)^2)
  ss_tot = sum((test.y - mean(test.y))^2)
  temperature.rsq[n] = 1 - ss_res / ss_tot

  cat(sprintf("Bootstrap %d\n", n))
  cat("Predictions (first 5):\n")
  print(round(head(preds, 5), 4))
  cat(sprintf("R² for Bootstrap %d: %.4f\n", n, temperature.rsq[n]))

  coef_vec = as.vector(coef(cv.fit, s = "lambda.min"))
  names(coef_vec) = rownames(coef(cv.fit, s = "lambda.min"))
  model.coefficients[[n]] = coef_vec

  results = foreach(i = 1:length(depthsToIterate), .combine = "c") %dopar% {
    tryCatch({
      if (!is.finite(depthsToIterate[i]) || is.na(depthsToIterate[i]) || depthsToIterate[i] <= 0) {
        bad.fids <<- c(bad.fids, gridCells$FID[i])
        return(NA)
      }

      mySlices = seq(1, depthsToIterate[i], 0.01)
      patch = data.frame(gridCells[i, ])
      patch = patch[rep(seq_len(nrow(patch)), each = length(mySlices)), ]
      patch$Depth = log10(mySlices)
      patch$temperature = gridCells[i, ]$mast + mySlices * gridCells[i, ]$gradient / 1000
      patch$dummy = 1

      patch = patch[, which(colnames(patch) %in% append(keep, "dummy"))]
      patch = model.matrix(dummy ~ ., patch)

      missing_cols = setdiff(colnames(train.x), colnames(patch))
      if (length(missing_cols) > 0) {
        tmp = matrix(0, nrow = nrow(patch), ncol = length(missing_cols))
        colnames(tmp) = missing_cols
        patch = cbind(patch, tmp)
      }

      patch = patch[, colnames(train.x), drop = FALSE]

      if (ncol(patch) == 0 || nrow(patch) == 0) {
        bad.fids <<- c(bad.fids, gridCells$FID[i])
        return(NA)
      }

      sum(10 ^ predict(cv.fit, newx = patch, s = "lambda.min"))
    }, error = function(e) {
      bad.fids <<- c(bad.fids, gridCells$FID[i])
      return(NA)
    })
  }

  results[is.na(results)] <- 0
  cv.byGridResult[, n] = results * gridCells$grid_area_m2 * 100 * 100
  cv.biomass[n] = sum(cv.byGridResult[, n])
  temperature.error[n] = mean((preds - test.y)^2)
}

proc.time() - ptm

write.table(cv.biomass, file = "glm_depthtempZ122_Med_HF_cv.biomass.csv", sep = ",", row.names = FALSE, col.names = FALSE)
write.table(temperature.error, file = "glm_depthtempZ122_Med_HF_cv.error.csv", sep = ",", row.names = FALSE, col.names = FALSE)
write.table(temperature.rsq, file = "glm_depthtempZ122_Med_HF_cv_rsquare.csv", sep = ",", row.names = FALSE, col.names = FALSE)
write.table(cv.byGridResult, file = "glm_depthtempZ122_Med_HF_cvGridResult.csv", sep = ",", row.names = FALSE, col.names = FALSE)
write.csv(do.call(rbind, model.coefficients), "glmnet_bootstrap_coefficients.csv", row.names = FALSE)
write.csv(data.frame(FID = unique(bad.fids)), "bad_gridcells.csv", row.names = FALSE)
save(model.coefficients, file = "glmnet_bootstrap_coefficients.RData")

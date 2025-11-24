cond <- grepl('/hpc', getwd())
if (cond){
  source('path/to/config.R')
} else {
  source('path/to/config.R')
} ; rm(cond)
set.seed(1)
library(AUCell)
library(irlba)
library(xgboost)


slurm_array_task_id <- Sys.getenv('SLURM_ARRAY_TASK_ID') %>% as.numeric()

setwd(apoe_ccre_objects)
trp <- list.files(pattern = '8.5.normalize_trp___.*.rds$')[slurm_array_task_id]
cluster_id <- gsub('.*___|_trans_net_norm.rds', '', trp)
trp <- readRDS(trp)

# tf at least 10% expressed
setwd(rna_nebula_objects)
degs <- paste0('2.nebula_degs___', cluster_id, '.rds')
degs <- readRDS(degs)
genes <- degs$summary$gene
ix <- colnames(trp) %in% genes
trp <- trp[, ix]

tf_names <- colnames(trp)
trp <- lapply(1:ncol(trp),
             function(x){
               x = sort(trp[, x], decreasing = TRUE)
               x = x[1:1000]
               return(x)
             })
names(trp) <- tf_names


setwd(rna_nebula_objects)
seu <- readRDS(paste0('1.split_objects___', cluster_id, '.rds'))
seu <- NormalizeData(seu) %>% FindVariableFeatures() %>% ScaleData() %>% RunPCA()
counts <- seu[['rna']]$data
counts <- t(counts)
pcs <- seu[['pca']]@cell.embeddings

setwd(covariate_analysis_objects)
metadata <- readRDS("1.get_metadata___sample_and_cell_level.rds")
rownames(metadata) <- gsub('HCTZZW', 'hctzzw', rownames(metadata))
metadata <- metadata[colnames(seu), ]
metadata$diagnosis <- ifelse(metadata$diagnosis == 'load', 1, 0)
metadata$sex <- ifelse(metadata$sex == 'female', 1, 0)
metadata$apoe <- ifelse(metadata$apoe == 'e34', 1, 0)
for (j in 1:ncol(metadata)){
  v <- metadata[, j]
  v <- gsub(' ', '', v)
  v <- gsub('HCTZZW|hctzzw', '123456', v)
  metadata[, j] <- as.numeric(v)
}
metadata$sample_id <- as.character(metadata$sample_id)
metadata <- cbind(metadata, as.data.frame(pcs))

# calculate eigengene for each TF
X <- NULL
for (i in 1:length(trp)){
  message('calculating eigengene ', i, ' of ', length(trp))
  trp_i = trp[[i]]
  x <- counts[, names(trp_i)]
  x <- x[, colSums(x) > 0]
  trp_i = trp_i[colnames(x)]
  
  trp_i = trp_i / mean(trp_i) # scale to mean of 1
  
  D = diag(sqrt(trp_i))
  D = as(D, 'dgCMatrix')
  x = scale(x)
  x = x %*% D # scale gene variances to respective trp
  x <- irlba(x, nv = 1)
  x <- as.numeric(scale(x$u))
  X[[i]] <- x
}
X <- do.call(what = cbind, args = X)

Y = metadata$diagnosis
metadata = metadata[, c('ncount_rna', 'nfeature_rna', 'seq_sat', 'age', 'sex', 'pmi', 'apoe', 'PC_1', 'PC_2', 'PC_3')]
metadata = as.matrix(metadata)

# generate knockoff variables
X_knockoff = NULL
for (j in 1:length(trp)){
  message('generating knockoff ', j, ' of ', length(trp))
  params <- list(
    objective = "reg:squarederror",
    # subsample = 0.8,           # Random subset each boosting step
    colsample_bytree = 0.5      # Random feature subsample too, optional
  )
  Xj = X[, j]
  Xp = X[, -j]
  
  Xp = cbind(Xp, metadata)
  
  idx <- sample(1:nrow(X), size = 0.8 * nrow(X))
  dtrain <- xgb.DMatrix(data = Xp[idx, ], label = Xj[idx])
  dvalid <- xgb.DMatrix(data = Xp[-idx, ], label = Xj[-idx])
  dall <- xgb.DMatrix(data = Xp, label = Xj)
  watchlist <- list(train = dtrain, eval = dvalid)
  
  conditional_u <- xgb.train(params = params,
                             data = dtrain, 
                             nrounds = 1000, 
                             max_depth = 6,
                             early_stopping_rounds = 25, 
                             watchlist = watchlist, 
                             verbose = 0)
  
  u <- predict(conditional_u, newdata = dall)
  
  dtrain2 <- xgb.DMatrix(data = Xp[idx, ], label = (Xj[idx] - u[idx])^2)
  dvalid2 <- xgb.DMatrix(data = Xp[-idx, ], label = (Xj[-idx] - u[-idx])^2)
  dall2 <- xgb.DMatrix(data = Xp, label = (Xj-u)^2)
  watchlist2 <- list(train = dtrain2, eval = dvalid2)
  
  conditional_sigma <- xgb.train(params = params, 
                                 data =dtrain2, 
                                 nrounds = 1000, 
                                 max_depth = 6, 
                                 early_stopping_rounds = 25, 
                                 watchlist = watchlist2,
                                 verbose = 0)
  conditional_sigma <- predict(conditional_sigma, newdata = dall2)
  sigma <- sqrt(pmax(conditional_sigma, 1e-6))
  
  fake <- u + rnorm(nrow(X), sd = sigma)
  
  # make sure looks ok
  # plot(u, Xj) # real data
  # points(u, fake, col = 'red') # simulated data
  
  X_knockoff[[j]] <- fake
}
X_knockoff <- do.call(what = cbind, args = X_knockoff)

# verify swap property

# calculate shapleys
W <- numeric()
X = cbind(X, metadata)
for (j in 1:length(trp)){
  message('calculating W statistic ', j, ' of ', length(trp))
  param <- list(
    objective = "binary:logistic",
    subsample = 0.8,         # Random subset each boosting step
    colsample_bytree = 0.5      # Random feature subsample too, optional
  )
  
  # observed test statistic
  idx <- sample(1:nrow(X), size = 0.8 * nrow(X))
  dtrain <- xgb.DMatrix(data = X[idx, ], label = Y[idx])
  dvalid <- xgb.DMatrix(data = X[-idx, ], label = Y[-idx])
  watchlist <- list(train = dtrain, eval = dvalid)
  dall = xgb.DMatrix(data = X, label = Y)
  
  model <- xgb.train(
    params = param,
    data = dtrain,
    nrounds = 1000,
    watchlist = watchlist,
    max_depth = 6,
    early_stopping_rounds = 25,   # Stop if eval doesn't improve for 10 rounds
    eval_metric = "logloss",
    maximize = FALSE,
    verbose = 0
  )
  
  shap <- predict(model, newdata = dall, predcontrib = TRUE)
  t_obs <- mean(abs(shap[,j]))
  
  Xswap <- X
  Xswap[, j] <- X_knockoff[, j]
  
  # knockoff test statistic
  dtrain2 <- xgb.DMatrix(data = Xswap[idx, ], label = Y[idx])
  dvalid2 <- xgb.DMatrix(data = Xswap[-idx, ], label = Y[-idx])
  watchlist2 <- list(train = dtrain2, eval = dvalid2)
  dall2 = xgb.DMatrix(data = Xswap, label = Y)
  
  model2 <- xgb.train(
    params = param,
    data = dtrain2,
    nrounds = 1000,
    watchlist = watchlist2,
    max_depth = 6,
    early_stopping_rounds = 25,   # Stop if eval doesn't improve for 10 rounds
    eval_metric = "logloss",
    verbose = 0
  )
  
  shap2 <- predict(model2, newdata = dall2, predcontrib = TRUE)
  t_knockoff <- mean(abs(shap2[,j]))
  
  W[j] = t_obs - t_knockoff
  
  if (length(W) > 1){
    plot(density(W))
  }
  
}
df <- data.frame(cluster_id = cluster_id, tf = names(trp), W = W)

setwd(apoe_ccre_objects)
saveRDS(df, paste0('9.model_x_knockoff___', cluster_id, '.rds'))

sesh <- capture.output(sessionInfo())
print(sesh)


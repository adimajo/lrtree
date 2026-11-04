# Exports the real datasets of glmtree's benchmark (inst/benchmarks/common.R) to gzipped CSV files, so that the
# Python benchmark does not need R. Run once:  Rscript benchmarks/export_datasets.R benchmarks/data
#
# Required packages: scorecard, MASS, ISLR, AER. Each file has the features and a 0/1 column `y`; factors are
# written as text. The datasets are the ones used by `../glmtree/inst/benchmarks/real_data.R`.

outdir <- if (length(commandArgs(TRUE))) commandArgs(TRUE)[1L] else "benchmarks/data"
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

load_data <- function(name, package) {
  environment <- new.env()
  utils::data(list = name, package = package, envir = environment)
  environment[[name]]
}
as_binary <- function(x, positive) as.integer(x == positive)

german <- load_data("germancredit", "scorecard")
pima <- rbind(load_data("Pima.tr", "MASS"), load_data("Pima.te", "MASS"))
default <- load_data("Default", "ISLR")
credit_card <- load_data("CreditCard", "AER")
caravan <- load_data("Caravan", "ISLR")

datasets <- list(
  german = cbind(german[, setdiff(names(german), "creditability")], y = as_binary(german$creditability, "bad")),
  pima = cbind(pima[, setdiff(names(pima), "type")], y = as_binary(pima$type, "Yes")),
  default = cbind(default[, c("student", "balance", "income")], y = as_binary(default$default, "Yes")),
  # share and expenditure leak the outcome
  creditcard = cbind(credit_card[, setdiff(names(credit_card), c("card", "share", "expenditure"))],
                     y = as_binary(credit_card$card, "yes")),
  caravan = cbind(caravan[, setdiff(names(caravan), "Purchase")], y = as_binary(caravan$Purchase, "Yes"))
)
for (name in names(datasets)) {
  data <- datasets[[name]]
  for (j in seq_along(data)) if (is.factor(data[[j]]) || is.logical(data[[j]])) data[[j]] <- as.character(data[[j]])
  file <- gzfile(file.path(outdir, paste0(name, ".csv.gz")), "w")
  utils::write.csv(data, file, row.names = FALSE)
  close(file)
  message(name, ": ", nrow(data), " rows, ", ncol(data) - 1L, " features, ", round(mean(data$y), 3), " positives")
}

# Step 4.3 (R part): the protocol's mixed negative binomial model on the SYNTHETIC WP2 data.
# A pipeline test, not findings. Steps per half-hour ~ green view (standardised) + controls,
# random intercept per person, glmmTMB nbinom2.
#   A: no environmental effect was generated, so the green-view coefficient should be near 0.
#   B: a planted effect of log(1.15) per SD of the TRUE green view was generated.
# The generator builds the step rate additively from minutes walking, cycling and staying (a park
# stay has a high rate), so no log-linear adjustment is exactly right. The run reports a
# specification curve instead of picking a winner after the fact: hour and weekend only; + walking
# and cycling minutes; + park-stay minutes (linear, log1p, and binned). Moderation by income stratum
# and precision by sample size use the walking-and-cycling specification, fixed before this run
# because park-stay minutes carry part of the green-view exposure.
# Usage: Rscript scripts/52_wp5_nbmm.R
suppressPackageStartupMessages(library(glmmTMB, lib.loc = Sys.getenv("R_LIBS_USER")))
set.seed(20261011)
d <- read.csv("outputs/wp5/model_input_halfhours.csv")
d$hour <- factor(d$hour)
d$stratum <- factor(d$stratum, levels = c("high", "middle", "low"))
bin <- function(x) cut(x, c(-1, 0, 5, 15, 31), labels = c("0", "0-5", "5-15", "15-30"))
d$walk_bin <- bin(d$walk_min); d$bike_bin <- bin(d$bike_min); d$park_bin <- bin(d$park_min)
beta_true <- log(1.15)
SPECS <- c("hour, weekend" = "",
           "+ walking, cycling minutes" = "+ walk_min + bike_min",
           "+ park-stay minutes" = "+ walk_min + bike_min + park_min",
           "+ all minutes, log(1 + x)" = "+ log1p(walk_min) + log1p(bike_min) + log1p(park_min)",
           "+ all minutes, binned" = "+ walk_bin + bike_bin + park_bin")

fit_one <- function(y, x, spec, data) {
  f <- paste(y, "~", x, "+ hour + weekend", SPECS[[spec]], "+ (1 | person_id)")
  m <- glmmTMB(as.formula(f), family = nbinom2, data = data)
  s <- summary(m)$coefficients$cond
  data.frame(outcome = y, exposure = x, adjusted = spec, estimate = s[x, "Estimate"], se = s[x, "Std. Error"],
             ci_lo = s[x, "Estimate"] - 1.96 * s[x, "Std. Error"], ci_hi = s[x, "Estimate"] + 1.96 * s[x, "Std. Error"],
             rate_ratio = exp(s[x, "Estimate"]), n = nrow(data), persons = length(unique(data$person_id)),
             converged = m$fit$convergence == 0, dispersion = sigma(m), aic = AIC(m))
}

res <- list()
for (y in c("steps_A_null", "steps_B_planted")) for (s in names(SPECS)) res[[length(res) + 1]] <- fit_one(y, "z_green", s, d)
res[[length(res) + 1]] <- fit_one("steps_B_planted", "z_true", "+ walking, cycling minutes", d)
res[[length(res) + 1]] <- fit_one("steps_B_planted", "z_true", "+ park-stay minutes", d)
res <- do.call(rbind, res)
res$beta_planted <- ifelse(res$outcome == "steps_B_planted", beta_true, 0)
res$SYNTHETIC <- TRUE
write.csv(res, "outputs/wp5/nbmm_results_synthetic.csv", row.names = FALSE)
print(res[, c("outcome", "exposure", "adjusted", "estimate", "ci_lo", "ci_hi", "aic")], digits = 3)

mod <- glmmTMB(steps_B_planted ~ z_green * stratum + hour + weekend + walk_min + bike_min + (1 | person_id),
               family = nbinom2, data = d)
mc <- as.data.frame(summary(mod)$coefficients$cond)
mc <- mc[grepl("z_green", rownames(mc)), ]
mc$term <- rownames(mc)
write.csv(mc, "outputs/wp5/nbmm_moderation_synthetic.csv", row.names = FALSE)
print(mc, digits = 3)

ids <- unique(d$person_id)
sub <- list()
for (n in c(50, 100, 150)) for (r in 1:20) {
  keep <- sample(ids, n)
  s <- fit_one("steps_B_planted", "z_green", "+ walking, cycling minutes", d[d$person_id %in% keep, ])
  s$subset_persons <- n; s$rep <- r
  sub[[length(sub) + 1]] <- s
}
sub <- do.call(rbind, sub)
agg <- do.call(rbind, lapply(split(sub, sub$subset_persons), function(g) data.frame(
  persons = g$subset_persons[1], mean_estimate = mean(g$estimate), sd_estimate = sd(g$estimate),
  mean_ci_width = mean(g$ci_hi - g$ci_lo), share_ci_excludes_0 = mean(g$ci_lo > 0),
  share_ci_covers_planted = mean(g$ci_lo <= beta_true & g$ci_hi >= beta_true))))
write.csv(sub, "outputs/wp5/nbmm_subsamples_synthetic.csv", row.names = FALSE)
write.csv(agg, "outputs/wp5/nbmm_subsample_summary_synthetic.csv", row.names = FALSE)
print(agg, digits = 3)
cat("NBMM_DONE\n")

#TN=4376
#FN=274
#TP=437
#FP=96

TN=492
FN=47
TP=451
FP=37

TNR=TN/(TN+FP) #Specificity
FPR=FP/(TN+FP) 
TPR=TP/(TP+FN) #Recall #Sensitivity
FNR=FN/(TP+FN)
Precission=TP/(TP+FP)
Total=TN+FP+TP+FN

# Binomial
binom_test_Accuracy = binom.test(TP+TN, Total, conf.level = 0.99)
binom_test_FN = binom.test(TP, TP+FN, p = 1-TPR, conf.level = 0.99)
binom_test_Specificity = binom.test(TN, TN+FP, p = 1-(TN+FP) / Total, conf.level = 0.99)
binom_test_Recall = binom.test(TPR*(TP + FN), (TP + FN), conf.level = 0.99)
binom_test_Accuracy
binom_test_FN
binom_test_Recall
binom_test_Specificity

# Udregning af precision, recall, and F1 score
precision <- TP / (TP + FP)
recall <- TP / (TP + FN)
f1_score <- 2 * (precision * recall) / (precision + recall)
f1_score
# Function to calculate confidence interval for F1 score using delta method
calculate_f1_ci <- function() {
  # Udregning af weights baseret på class frequency
  weight_precision <- (TP + FP) / Total
  weight_recall <- (TP + FN) / Total
  
  # Standard error
  se_precision <- sqrt(weight_precision * (1 - precision) / Total)
  se_recall <- sqrt(weight_recall * (1 - recall) / Total)
  
  # Delta method for F1score standard error
  
  se_f1_score <- sqrt((se_precision^2 * recall^2 + se_recall^2 * precision^2) / (precision + recall)^4)

  # udregning af 95% CI
  lower_bound <- f1_score - qnorm(0.995) * se_f1_score
  upper_bound <- f1_score + qnorm(0.995) * se_f1_score
  
  # Calculate p-value
  z_stat <- (f1_score - 0.5) / se_f1_score
  p_value <- 2 * (1 - pnorm(abs(z_stat)))
  print(se_f1_score)
  cat("F1 Score Confidence Interval:\n")
  print(c(lower_bound, upper_bound))
  cat("p-value:\n")
  print(p_value)
  print(pnorm(abs(z_stat)))
}

calculate_f1_ci()

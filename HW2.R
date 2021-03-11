library(ggplot2)
library(tidyverse)
library(rsample)
library(caret)
library(modelr)
library(parallel)
library(foreach)
library(mosaic)
library(jtools)
library(dbplyr)
library(naivebayes)

library(gamlr)

#Question 1

capmetro = read.csv('/Users/franklinstudent/Desktop/GitHub/Exercise-2/capmetro_UT.csv')


# Recode the categorical variables in sensible, rather than alphabetical, order
capmetro = mutate(capmetro,
                     day_of_week = factor(day_of_week,
                                          levels=c("Mon", "Tue", "Wed","Thu", "Fri", "Sat", "Sun")),
                     month = factor(month,
                                    levels=c("Sep", "Oct","Nov")))


months = c('Sep', 'Oct', 'Nov')

count_boardings = capmetro %>%
  filter(month %in% months) %>%
  group_by(hour_of_day, day_of_week, month) %>%
  summarize(total_boardings = n(),
            boarding = sum(boarding),
            average_boardings = boarding/total_boardings)

bar_graph = ggplot(count_boardings) + 
  geom_line(aes(x = hour_of_day, y = average_boardings, color = month)) + 
  facet_wrap(~day_of_week) + scale_x_continuous(name = "Hour of Day") + 
  scale_y_continuous(name = "Average Boardings") + ggtitle("Capital Metro") +
  theme(plot.title = element_text(hjust = 0.5)) + labs(color = "Month")

bar_graph

# Excluding the weekends, peak boardings are broadly similar, which is during evening 
# rush hour. 

# The average boardings on Monday are lower in September because of Labor Day. 
# Since its a national holiday, the university is closed and UT students would 
# stay home ana would not need to use the bus. 



# Additionally, average boarding is lower on Wed/Thurs/Fri for November 
# because Thanksgiving lands on a Thursday in November, and UT students would most 
# likely not travel during days of that particular week, but instead spending time 
# with family/friends. 




ggplot(data = capmetro) + 
  geom_point(mapping = aes(x = temperature, y = boarding, color = weekend)) + 
  facet_wrap(~hour_of_day) + ggtitle("Capitol Metro") +
  theme(plot.title = element_text(hjust = 0.5)) + scale_x_continuous(name = "Temperature") +
  scale_y_continuous(name = "Boardings") + labs(color = "Day Type")
  
# When holding both hour of day and weekend status constant, it does not appear that 
# temperature has a noticeable effect on the number of UT students riding the bus. This
# is determined by the observable changes in ridership throughout the day. Within each hour
# of the day, the quantity of boardings remain similar as the temperature changes. 






#Question 2

data(SaratogaHouses)

#Standardize
Houses_scale = SaratogaHouses %>%
  mutate(across(lotSize,scale)) %>%
  mutate(across(age,scale)) %>%
  mutate(across(landValue,scale)) %>%
  mutate(across(livingArea,scale)) %>%
  mutate(across(bedrooms,scale)) %>%
  mutate(across(fireplaces,scale)) %>%
  mutate(across(bathrooms,scale)) %>%
  mutate(across(rooms,scale)) %>%
  mutate(across(pctCollege,scale)) 



saratoga_split = initial_split(Houses_folds, prop = 0.8)
saratoga_train = training(saratoga_split)
saratoga_test = testing(saratoga_split)


lm2 = lm(price ~ . - pctCollege - sewer - waterfront - landValue - newConstruction, data=saratoga_train)
lm0 = lm(price ~ 1, data = saratoga_train)
lm_forward = step(lm0, direction = 'forward',
      scope=~(lotSize + age + landValue + livingArea + bedrooms + 
                bathrooms + rooms + waterfront + centralAir + newConstruction + heating))

AIC(lm2)
AIC(lm_forward)

summary(lm2)
summary(lm_forward)

getCall(lm_forward)
coef(lm_forward)






rmse_lm2 = rmse(lm2, saratoga_test)
rmse_lm_forward = rmse(lm_forward, saratoga_test)

rmse_lm2
rmse_lm_forward

knn_lm_forward = knnreg(price ~ livingArea + landValue + bathrooms + waterfront + 
         newConstruction + heating + centralAir + lotSize + bedrooms + 
         rooms + age, data = Houses_scale, k = 20)

K_folds = 30
k_grid = seq(2, 80, by=2)
Houses_folds = crossv_kfold(Houses_scale, k=K_folds)

cv_grid = foreach(k = k_grid, .combine='rbind') %do% {
  models = map(Houses_folds$train, ~ knnreg(price ~ livingArea + landValue + bathrooms + waterfront + 
                                            newConstruction + centralAir + heating + lotSize + bedrooms + 
                                            rooms + age, k=k, data = Houses_scale, use.all=FALSE))
  errs = map2_dbl(models, Houses_folds$test, modelr::rmse)
  c(k=k, err = mean(errs), std_err = sd(errs)/sqrt(K_folds))
} %>% as.data.frame


ggplot(cv_grid) +
  geom_point(aes(x=k, y=err)) +
  geom_errorbar(aes(x=k, ymin = err-std_err, ymax = err+std_err)) + 
  labs(y="RMSE", title="RMSE vs k for KNN regression")




knnlm2 = knnreg(price ~ . - pctCollege - sewer - waterfront - landValue - newConstruction,
       data = Houses_scale, k = 20)

cv_grid = foreach(k = k_grid, .combine='rbind') %do% {
  models = map(Houses_folds$train, ~ knnreg(price ~ . - pctCollege - sewer - waterfront - 
                                              landValue - newConstruction,
                                            k=k, data = Houses_scale, use.all=FALSE))
  errs = map2_dbl(models, Houses_folds$test, modelr::rmse)
  c(k=k, err = mean(errs), std_err = sd(errs)/sqrt(K_folds))
} %>% as.data.frame


ggplot(cv_grid) +
  geom_point(aes(x=k, y=err)) +
  geom_errorbar(aes(x=k, ymin = err-std_err, ymax = err+std_err)) + 
  labs(y="RMSE", title="RMSE vs k for KNN regression")



















# Question 3

german_credit = read.csv('/Users/franklinstudent/Desktop/GitHub/Exercise-2/german_credit.csv')

Credit_Rating = c('good', 'poor', 'terrible')

count_history = german_credit %>%
  filter(history %in% Credit_Rating) %>%
  group_by(history)%>%
  summarize(total_history = n(),
            default = sum(Default == 1),
            Probability = default/total_history)

ggplot(data = count_history) + 
  geom_col(mapping = aes(x = Credit_Rating, y = Probability, fill = Credit_Rating), 
           position = 'dodge') + ggtitle("Default Probabilities") +
  theme(plot.title = element_text(hjust = 0.5))


glm = glm(Default ~ duration + amount + installment + age + history + 
            purpose + foreign, data = german_credit, family = 'binomial')


coef(glm)

# It should be noted that the default probabilities are much lower for individuals with 
# either a poor or terrible credit history relative to individuals with a good 
# history. Although, when observing the glm, individuals with either a 
# poor or terrible credit history have a negative effect in regards to defaulting
# on a loan. I believe that these conflicting observations are due to the bank
# having a more strict loan process for individuals with either a poor or terrible 
# credit history. Individuals that are in either category are more likely to repay 
# their loans back on time. In contrast, indivuals with a good credit history are
# given a more lenient loan process and as a result, have a higher probability in 
# defaulting on their loans. 

# If the purpose of the model is to screen prospective borrowers to classify them 
# into "high" versus "low" probability of default, then no, I do not believe this 
# data set accomplished its goal. The conclusion from this data set should not be
# to loan less to individuals to a good credit history. Instead, the end result 
# should be for the bank to apply the same strict loan standards that are applied 
# to others that may have either a poor or terrible credit history. This would give
# a better source of data and better inform the bank on the probability of default.











# Question 4

dev = read_csv('/Users/franklinstudent/Desktop/GitHub/Exercise-2/hotels_dev.csv')


dev_split = initial_split(dev, prop = 0.8)
dev_train = training(dev_split)
dev_test = testing(dev_split)


lm_dev1 = lm(children ~ market_segment + adults + customer_type + is_repeated_guest, data = dev_train)
lm_dev2 = lm(children ~ . - arrival_date, data = dev_train)
lm_devbest = lm(children ~ hotel + lead_time+ reserved_room_type + assigned_room_type + booking_changes + adults + 
                  required_car_parking_spaces + booking_changes + average_daily_rate + is_repeated_guest + arrival_date, data = dev_train)



#Baseline 1: Small Model
phat_train_dev1 = predict(lm_dev1, dev_train)
yhat_train_dev1 = ifelse(phat_train_dev1 > 0.5, 1, 0)
confusion_in1 = table(y = dev_train$children, yhat = yhat_train_dev1)
confusion_in1
sum(diag(confusion_in1))/sum(confusion_in1)

phat_test_dev1 = predict(lm_dev1, dev_test)
yhat_test_dev1 = ifelse(phat_test_dev1 > 0.5, 1, 0)
confusion_out1 = table(y = dev_test$children, yhat = yhat_test_dev1)
confusion_out1
sum(diag(confusion_out1))/sum(confusion_out1)


#Baseline 2: Big Model 
phat_train_dev2 = predict(lm_dev2, dev_train)
yhat_train_dev2 = ifelse(phat_train_dev2 > 0.5, 1, 0)
confusion_in2 = table(y = dev_train$children, yhat = yhat_train_dev2)
confusion_in2
sum(diag(confusion_in2))/sum(confusion_in2)

phat_test_dev2 = predict(lm_dev2, dev_test)
yhat_test_dev2 = ifelse(phat_test_dev2 > 0.5, 1, 0)
confusion_out2 = table(y = dev_test$children, yhat = yhat_test_dev2)
confusion_out2
sum(diag(confusion_out2))/sum(confusion_out2)


#Best Linear Model
phat_train_devbest = predict(lm_devbest, dev_train)
yhat_train_devbest = ifelse(phat_train_devbest > 0.5, 1, 0)
confusion_in_devbest = table(y = dev_train$children, yhat = yhat_train_devbest)
confusion_in_devbest
sum(diag(confusion_in_devbest))/sum(confusion_in_devbest)


phat_test_devbest = predict(lm_devbest, dev_test)
yhat_test_devbest = ifelse(phat_test_devbest > 0.5, 1, 0)
confusion_out_devbest = table(y = dev_test$children, yhat = yhat_test_devbest)
confusion_out_devbest
sum(diag(confusion_out_devbest))/sum(confusion_out_devbest)

table(dev_train$children)
33096/sum(table(dev_train$children))


table(dev_test$children)
8269/sum(table(dev_test$children))


#abolute improvement 
0.9193078 - 0.9188799


#The relative improvement
0.9193078/0.9188799




val = read_csv('/Users/franklinstudent/Desktop/GitHub/Exercise-2/hotels_val.csv')


logit_val = glm(children ~ hotel + lead_time+ reserved_room_type + assigned_room_type + 
                  booking_changes + adults + required_car_parking_spaces + booking_changes + 
                  average_daily_rate + is_repeated_guest + arrival_date, data = val, family = 'binomial')

coef(logit_val)


phat_test_logit_val = predict(logit_val, dev_test, type = 'response')
yhat_test_logit_val = ifelse(phat_test_logit_val > 0.5, 1, 0)
confusion_out_logit= table(y = dev_test$children, yhat = yhat_test_logit_val)
confusion_out_logit


#Error Rate of 6.5%
(510 + 75)/8999

#Accuracy Rate of 93.49%
1 - (510 + 75)/8999

#Lift over LPM: 1.02
0.9349928/0.9193078

#True Positive Rate (TPR) of 30.14%
220/(510 + 220)

#False Positive Rate (FPR) of 0.9%
75/ (8194 + 75)

#False Discovery Rate (FDR) of 24.42% 
75/(75+220)


#Model Validation 1
phat_test_logit_val = predict(logit_val, dev_test, type = 'response')
thresh_grid = seq(0.95, 0.05, by=-0.005)


roc_curve = foreach(thresh = thresh_grid, .combine='rbind') %do% {
  yhat_test_logit_val = ifelse(phat_test_logit_val >= thresh, 1, 0)
  confusion_out_logit = table(children = dev_test$children, yhat = yhat_test_logit_val)
  out_logit = data.frame(model = "logit",
                         TPR = confusion_out_logit[2,2]/sum(dev_test$children==1),
                         FPR = confusion_out_logit[1,2]/sum(dev_test$children==0))
  
  rbind(out_logit)
} %>% as.data.frame()

roc = ggplot(roc_curve) + 
  geom_line(aes(x=FPR, y=TPR, color=model)) + 
  labs(title="ROC curve: Logit Model") +
  theme_bw(base_size = 10)

roc 



#Model Validation 2


X_NB = as.matrix(val)
Y_NB = factor(val$children)

N = length(Y_NB)
train_frac = 0.8
train_set = sample.int(N, floor(train_frac*N)) %>% sort
test_set = setdiff(1:N, train_set)


X_train = X_NB[train_set,]
X_test = X_NB[test_set,]

Y_train = Y_NB[train_set]
Y_test = Y_NB[test_set]

nb_model = multinomial_naive_bayes(x = X_train, y = Y_train)

y_test_pred = predict(nb_model, X_test)

table(y_test, y_test_pred)

sum(diag(table(Y_test, y_test_pred)))/length(Y_test)






knn_lm_forward = knnreg(price ~ livingArea + landValue + bathrooms + waterfront + 
                          newConstruction + heating + centralAir + lotSize + bedrooms + 
                          rooms + age, data = Houses_scale, k = 20)

K_folds = 30
k_grid = seq(2, 80, by=2)
Houses_folds = crossv_kfold(Houses_scale, k=K_folds)



library(ggplot2)
library(tidyverse)
library(rsample)
library(caret)
library(modelr)
library(parallel)
library(foreach)
library(mosaic)
library(jtools)

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

saratoga_split = initial_split(SaratogaHouses, prop = 0.8)
saratoga_train = training(saratoga_split)
saratoga_test = testing(saratoga_split)


lm2 = lm(price ~ . - pctCollege - sewer - waterfront - landValue - newConstruction, data=saratoga_train)
lm0 = lm(price ~ 1, data = saratoga_train)
lm_forward = step(lm0, direction = 'forward',
      scope=~(pctCollege + landValue + bedrooms + fireplaces + bathrooms + waterfront
      + heating + fuel))

lm2



k_grid = seq(2, 80, by=2)


cv_grid = foreach(k = k_grid, .combine='rbind') %do% {
  models = map(lm_forward$train, ~ knnreg(price ~ landValue + 
          bathrooms + bedrooms + waterfront + fireplaces + 
          heating, k=k, data = ., use.all=FALSE))
  errs = map2_dbl(models, lm_forward$test, modelr::rmse)
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


summ(glm)

# Notably, the default probabilities are much lower for individuals with 
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

val = read.csv('/Users/franklinstudent/Desktop/GitHub/Exercise-2/hotels_val.csv')




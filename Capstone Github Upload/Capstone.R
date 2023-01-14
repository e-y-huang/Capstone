##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################
# Note: This code may take a long time to run.
# The vast majority of the run-time is regularization.
# To just run to verify final RMSE, first run the code up to line 851.
# Skip lines 852-925, which take a long time to run.
# Then run lines 926 to the end.
Sys.time()
if (!require("tidyverse")) install.packages("tidyverse")
if (!require("caret")) install.packages("caret")
if (!require("data.table")) install.packages("data.table")
if (!require("xgboost")) install.packages("xgboost")
library(tidyverse)
library(caret)
library(data.table)
library(lubridate)
library(xgboost)

# Creating folder to output RDA files.
dir.create("./Data/", showWarnings = F)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip
#
# Backup website:
# http://cse-grouplens-prd-web.oit.umn.edu/datasets/movielens/10m
# http://cse-grouplens-prd-web.oit.umn.edu/datasets/movielens/ml-10m.zip

url <- "https://files.grouplens.org/datasets/movielens/ml-10m.zip"
backup_url <- "http://cse-grouplens-prd-web.oit.umn.edu/datasets/movielens/ml-10m.zip"

dl <- tempfile()
download.file(url, dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))
movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind = 'Rounding')
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(backup_url, url, dl, ratings, movies, test_index, temp, movielens, removed)
Sys.time()

######
# Vector of genres noted in the README file.
# https://files.grouplens.org/datasets/movielens/ml-10m-README.html
######
genrelist <- c("Action", "Adventure", "Animation", 
               "Children", "Comedy", "Crime", 
               "Documentary", "Drama", "Fantasy",
               "Film-Noir", "Horror", "Musical",
               "Mystery", "Romance", "Sci-Fi",
               "Thriller", "War", "Western")
######
# release_year = year extracted from title
# review_year = year extracted from timestamp at time of review
# genres separated into columns
######
prepare_data <- function(data){
  new_data <- data %>% mutate_at(c("userId","movieId"), as.factor)
  new_data <- new_data %>%
    mutate(release_year = as.numeric(str_sub(title, start=-5L,end=-2L)),
           review_year = year(as_datetime(timestamp))) %>%
    select(-title, -timestamp)
  data_genres <- as_tibble(sapply(genrelist, function(genre){
    str_detect(new_data$genres, genre)
  }))
  new_data <- new_data %>% 
    mutate(data_genres) %>%
    select(-genres) %>%
    as_tibble()
  return(new_data)
}

######
# Preparing non-validation data.
######
prep_edx <- prepare_data(edx)

######
# Plots used in report.
######
movie_info <- prep_edx %>%
  group_by(movieId) %>%
  mutate(avg_rating = mean(rating), movie_ratings = n()) %>%
  ungroup() %>%
  select(movieId, avg_rating, movie_ratings, release_year, all_of(genrelist)) %>%
  distinct(movieId, .keep_all=TRUE) %>%
  arrange(movieId)

genre_count <- prep_edx %>%
  select(all_of(genrelist)) %>% 
  colSums()

movie_genre_count <- movie_info %>%
  select(all_of(genrelist)) %>% 
  colSums()

cors <- prep_edx %>%
  select(-userId, -movieId, -rating) %>%
  sapply(cor, x = prep_edx$rating) %>%
  sapply(abs) %>%
  sort(decreasing = TRUE)

movie_cors <- movie_info %>%
  select(-movieId, -avg_rating) %>%
  sapply(cor, x = movie_info$avg_rating) %>%
  sapply(abs) %>%
  sort(decreasing = TRUE)

release_plot <- movie_info %>%
  ggplot(aes(release_year, avg_rating)) + 
  geom_point(alpha = 0.1) + 
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs")) + 
  labs(caption = "Release Year vs Average Rating by Movie")

movie_ratings_plot <- movie_info %>%
  ggplot(aes(movie_ratings, avg_rating)) +
  geom_point(alpha = 0.1) + 
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs")) +
  scale_x_sqrt() + 
  labs(caption = "Number of Movie Ratings vs Average Rating by Movie")

user_ratings_plot <- prep_edx %>%
  group_by(userId) %>%
  summarize(user_ratings = n(), avg_rating = mean(rating)) %>%
  ggplot(aes(user_ratings, avg_rating)) + 
  geom_point(alpha = 0.1) + 
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs")) +
  scale_x_sqrt() +
  labs(caption = "Number of User Ratings vs Average Rating by User")

drama_plot <- movie_info %>% 
  ggplot(aes(avg_rating, fill = `Drama`)) + 
  geom_density(alpha = 0.2) + 
  labs(caption = "Average Movie Ratings (Drama vs. Non-Drama)")

horror_plot <- movie_info %>% 
  ggplot(aes(avg_rating, fill = `Horror`)) + 
  geom_density(alpha = 0.2) + 
  labs(caption = "Average Movie Ratings (Horror vs. Non-Horror)")

fantasy_plot <- movie_info %>% 
  ggplot(aes(avg_rating, fill = `Fantasy`)) + 
  geom_density(alpha = 0.2) + 
  labs(caption = "Average Movie Ratings (Fantasy vs. Non-Fantasy)")

thriller_plot <- movie_info %>% 
  ggplot(aes(avg_rating, fill = `Thriller`)) + 
  geom_density(alpha = 0.2) + 
  labs(caption = "Average Movie Ratings (Thriller vs. Non-Thriller)")

######
# Creating 5 data partitions.
######
set.seed(2, sample.kind = "Rounding")
edx_indices <- createDataPartition(prep_edx$rating, times = 3, p = 0.1)

generate_sets <- function(dataset, indices){
  sets <- lapply(indices, function(index){
    train_set <- dataset %>% 
      filter(!(row_number() %in% index))
    temp <- dataset %>%
      filter(row_number() %in% index)
    
    test_set <- temp %>%
      semi_join(train_set, by = "movieId") %>%
      semi_join(train_set, by = "userId")
    
    removed <- temp %>%
      anti_join(test_set, by = c("movieId","userId"))
    train_set <- train_set %>% 
      rbind(removed)
    
    return(list(train = train_set, test = test_set))
  })
  return(sets)
}
sets <- generate_sets(dataset = prep_edx,
                      indices = edx_indices)
######
# Linear-based RMSEs.
######
# Adjusts by movieId and userId.
generate_basic_rmses <- function(setpairs = sets, l_m = 0, l_u = 0){
  sapply(setpairs, function(setpair){
    train_set <- setpair$train
    test_set <- setpair$test
    
    ######
    # Adjusting by mean rating for feature creation.
    ######
    mu <- train_set %>% pull(rating) %>% mean()
    train_set <- train_set %>%
      mutate(adj = mu)
    
    ######
    # Creating movie-related feature.
    ######
    movie_data <- train_set %>% 
      group_by(movieId) %>%
      summarize(b_m = sum(rating - adj)/(n() + l_m))
    train_set <- train_set %>%
      left_join(movie_data, by = "movieId") %>%
      mutate(adj = adj + b_m)
    
    ######
    # Creating user-related feature.
    ######
    user_data <- train_set %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - adj)/(n() + l_u))
    train_set <- train_set %>%
      left_join(user_data, by = "userId") %>%
      mutate(adj = adj + b_u)
    
    ######
    # Generating predictions and calculating RMSE.
    ######
    test_set <- test_set %>%
      mutate(adj = mu) %>%
      left_join(movie_data, by = "movieId") %>%
      left_join(user_data, by = "userId") %>%
      mutate(adj = adj + b_m + b_u)
    
    y_test <- test_set %>%
      pull(rating)
    y_pred <- test_set %>%
      pull(adj)
    
    score <- RMSE(y_pred, y_test)
    return(score)
  })
}

# Adjusts by movieId, userId, Drama, and Horror.
generate_g_rmses <- function(setpairs = sets, l_m = 0, l_u = 0){
  sapply(setpairs, function(setpair){
    train_set <- setpair$train
    test_set <- setpair$test
    
    ######
    # Adjusting by mean rating for feature creation.
    ######
    mu <- train_set %>% pull(rating) %>% mean()
    train_set <- train_set %>%
      mutate(adj = mu)
    
    ######
    # Creating movie-related features.
    ######
    drama_data <- train_set %>%
      group_by(Drama) %>%
      summarize(b_drama = mean(rating - adj))
    train_set <- train_set %>% 
      left_join(drama_data, by = "Drama") %>%
      mutate(adj = adj + b_drama)
    
    horror_data <- train_set %>%
      group_by(Horror) %>%
      summarize(b_horror = mean(rating - adj)) 
    train_set <- train_set %>%
      left_join(horror_data, by = "Horror") %>%
      mutate(adj = adj + b_horror)
    
    movie_data <- train_set %>% 
      group_by(movieId) %>%
      summarize(b_m = sum(rating - adj)/(n()+l_m))
    train_set <- train_set %>%
      left_join(movie_data, by = "movieId") %>%
      mutate(adj = adj + b_m)
    
    ######
    # Creating user-related feature.
    ######
    user_data <- train_set %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - adj)/(n()+l_u))
    train_set <- train_set %>%
      left_join(user_data, by = "userId") %>%
      mutate(adj = adj + b_u)
    
    ######
    # Generating predictions and calculating RMSE.
    ######
    test_set <- test_set %>%
      mutate(adj = mu) %>%
      left_join(drama_data, by = "Drama") %>%
      left_join(horror_data, by = "Horror") %>%
      left_join(movie_data, by = "movieId") %>%
      left_join(user_data, by = "userId") %>%
      mutate(adj = adj + b_drama + b_horror + b_m + b_u)
    
    y_test <- test_set %>%
      pull(rating)
    y_pred <- test_set %>%
      pull(adj)
    
    score <- RMSE(y_pred, y_test)
    return(score)
  })
}

# Adjusts by movieId, userId, and all genres.
generate_g_all_rmses <- function(setpairs = sets, l_m = 0, l_u = 0){
  sapply(setpairs, function(setpair){
    train_set <- setpair$train
    test_set <- setpair$test
    
    ######
    # Adjusting by mean rating for feature creation.
    ######
    mu <- train_set %>% pull(rating) %>% mean()
    train_set <- train_set %>%
      mutate(adj = mu)
    
    ######
    # Creating movie-related feature.
    ######
    movie_data <- train_set %>%
      group_by(`Action`) %>%
      mutate(b_g01 = mean(rating - adj), adj = adj + b_g01) %>%
      group_by(`Adventure`) %>%
      mutate(b_g02 = mean(rating - adj), adj = adj + b_g02) %>%
      group_by(`Animation`) %>%
      mutate(b_g03 = mean(rating - adj), adj = adj + b_g03) %>%
      group_by(`Children`) %>%
      mutate(b_g04 = mean(rating - adj), adj = adj + b_g04) %>%
      group_by(`Comedy`) %>%
      mutate(b_g05 = mean(rating - adj), adj = adj + b_g05) %>%
      group_by(`Crime`) %>%
      mutate(b_g06 = mean(rating - adj), adj = adj + b_g06) %>%
      group_by(`Documentary`) %>%
      mutate(b_g07 = mean(rating - adj), adj = adj + b_g07) %>%
      group_by(`Drama`) %>%
      mutate(b_g08 = mean(rating - adj), adj = adj + b_g08) %>%
      group_by(`Fantasy`) %>%
      mutate(b_g09 = mean(rating - adj), adj = adj + b_g09) %>%
      group_by(`Film-Noir`) %>%
      mutate(b_g10 = mean(rating - adj), adj = adj + b_g10) %>%
      group_by(`Horror`) %>%
      mutate(b_g11 = mean(rating - adj), adj = adj + b_g11) %>%
      group_by(`Musical`) %>%
      mutate(b_g12 = mean(rating - adj), adj = adj + b_g12) %>%
      group_by(`Mystery`) %>%
      mutate(b_g13 = mean(rating - adj), adj = adj + b_g13) %>%
      group_by(`Romance`) %>%
      mutate(b_g14 = mean(rating - adj), adj = adj + b_g14) %>%
      group_by(`Sci-Fi`) %>%
      mutate(b_g15 = mean(rating - adj), adj = adj + b_g15) %>%
      group_by(`Thriller`) %>%
      mutate(b_g16 = mean(rating - adj), adj = adj + b_g16) %>%
      group_by(`War`) %>%
      mutate(b_g17 = mean(rating - adj), adj = adj + b_g17) %>%
      group_by(`Western`) %>%
      mutate(b_g18 = mean(rating - adj), adj = adj + b_g18) %>%
      ungroup() %>%
      mutate(b_g = select(., b_g01:b_g18) %>% rowSums()) %>%
      group_by(movieId) %>%
      summarize(b_m = sum(rating - adj)/(n()+l_m) + mean(b_g))
    
    train_set <- train_set %>%
      left_join(movie_data, by = "movieId") %>%
      mutate(adj = adj + b_m)
    
    ######
    # Creating user-related feature.
    ######
    user_data <- train_set %>%
      group_by(userId) %>%
      summarize(b_u = sum(rating - adj)/(n()+l_u))
    train_set <- train_set %>%
      left_join(user_data, by = "userId") %>%
      mutate(adj = adj + b_u)
    
    ######
    # Generating predictions and calculating RMSE.
    ######
    test_set <- test_set %>%
      mutate(adj = mu) %>%
      left_join(movie_data, by = "movieId") %>%
      left_join(user_data, by = "userId") %>%
      mutate(adj = adj + b_m + b_u)
    
    y_test <- test_set %>%
      pull(rating)
    y_pred <- test_set %>%
      pull(adj)
    score <- RMSE(y_pred, y_test)
    return(score)
  })
}

######
# Values needed for xgbTree model.
######
# xgbTree uses entire test set and prints grid information.
control <- caret::trainControl(
  method = "none",
  verboseIter = TRUE,
  allowParallel = FALSE
)

# Default xgbTree grid values.
grid <- expand.grid(
  nrounds = 100, 
  max_depth = 6, 
  eta = 0.3, 
  gamma = 0,
  colsample_bytree = 1,
  min_child_weight = 1,
  subsample = 1
)

######
# Tree-based RMSEs.
# All models use release_year, review_year, movie_ratings, and user_ratings.
######

# Creates adj, b_m, and b_u just like the basic_rmses linear model.
# Does not use any genre data.
generate_basic_tree_rmses <- function(setpairs = sets, l_m = 0, l_u = 0){
  sapply(setpairs, function(setpair){
    train_set <- setpair$train
    test_set <- setpair$test
    
    ######
    # Adjusting by mean rating for feature creation.
    ######
    mu <- mean(train_set$rating)
    train_set <- train_set %>%
      mutate(adj = mu)
    
    ######
    # Creating movie-related features.
    ######
    movie_data <- train_set %>% 
      group_by(movieId) %>%
      summarize(movie_ratings = n(),
                b_m = sum(rating - adj)/(n()+l_m))
    train_set <- train_set %>%
      left_join(movie_data, by = "movieId") %>%
      mutate(adj = adj + b_m)
    
    ######
    # Creating user-related features.
    ######
    user_data <- train_set %>%
      group_by(userId) %>%
      summarize(user_ratings = n(), 
                b_u = sum(rating - adj)/(n()+l_u))
    train_set <- train_set %>%
      left_join(user_data, by = "userId") %>%
      mutate(adj = adj + b_u)
    
    ######
    # Training xgb model.
    ######
    x_train <- train_set %>% 
      select(release_year, review_year, adj,
             movie_ratings, b_m, 
             user_ratings, b_u) %>%
      sapply(as.numeric)
    y_train <- train_set %>%
      pull(rating)
    train_xgb <- caret::train(x_train, y_train, 
                              method = "xgbTree",
                              tuneGrid = grid,
                              trControl = control)
    
    ######
    # Generating predictions and calculating RMSE.
    ######
    print("Generating predictions.")
    test_set <- test_set %>%
      mutate(adj = mu) %>%
      left_join(movie_data, by = "movieId") %>%
      left_join(user_data, by = "userId") %>%
      mutate(adj = adj + b_m + b_u)
    
    x_test <- test_set %>% 
      select(release_year, review_year, adj,
             movie_ratings, b_m, 
             user_ratings, b_u) %>%
      sapply(as.numeric)
    y_test <- test_set %>%
      pull(rating)
    
    y_pred <- predict(train_xgb, x_test)
    score <- RMSE(y_pred, y_test)
    
    print("Done.")
    return(score)
  })
}

# Creates adj, b_m, and b_u just like the basic_rmses linear model.
# Uses Drama and Horror data after dealing with movie effect.
generate_g_tree_rmses <- function(setpairs = sets, l_m = 0, l_u = 0){
  sapply(setpairs, function(setpair){
    train_set <- setpair$train
    test_set <- setpair$test
    
    ######
    # Adjusting by mean rating for feature creation.
    ######
    mu <- mean(train_set$rating)
    train_set <- train_set %>%
      mutate(adj = mu)
    
    ######
    # Creating movie-related features.
    # Genre-related features are automatically dealt with by xgbTree.
    ######
    movie_data <- train_set %>% 
      group_by(movieId) %>%
      summarize(movie_ratings = n(),
                b_m = sum(rating - adj)/(n()+l_m))
    train_set <- train_set %>%
      left_join(movie_data, by = "movieId") %>%
      mutate(adj = adj + b_m)
    
    ######
    # Creating user-related features.
    ######
    user_data <- train_set %>%
      group_by(userId) %>%
      summarize(user_ratings = n(), 
                b_u = sum(rating - adj)/(n()+l_u))
    train_set <- train_set %>%
      left_join(user_data, by = "userId") %>%
      mutate(adj = adj + b_u)
    
    ######
    # Training xgb model.
    ######
    x_train <- train_set %>% 
      select(release_year, review_year, adj,
             movie_ratings, b_m, 
             user_ratings, b_u,
             Drama, Horror) %>%
      sapply(as.numeric)
    y_train <- train_set %>%
      pull(rating)
    
    train_xgb <- caret::train(x_train, y_train, 
                              method = "xgbTree",
                              tuneGrid = grid,
                              trControl = control)
    
    ######
    # Generating predictions and calculating RMSE.
    ######
    print("Generating predictions.")
    test_set <- test_set %>%
      mutate(adj = mu) %>%
      left_join(movie_data, by = "movieId") %>%
      left_join(user_data, by = "userId") %>%
      mutate(adj = adj + b_m + b_u)
    
    x_test <- test_set %>% 
      select(release_year, review_year, adj,
             movie_ratings, b_m, 
             user_ratings, b_u,
             Drama, Horror) %>%
      sapply(as.numeric)
    y_test <- test_set %>%
      pull(rating)
    
    y_pred <- predict(train_xgb, x_test)
    score <- RMSE(y_pred, y_test) 
    
    print("Done.")
    return(score)
  })
}

# Creates adj, b_m, and b_u just like the basic_rmses linear model.
# Calculates genre effect before dealing with movie effect.
generate_g_all_tree_rmses <- function(setpairs = sets, l_m = 0, l_u = 0){
  sapply(setpairs, function(setpair){
    train_set <- setpair$train
    test_set <- setpair$test
    
    ######
    # Adjusting by mean rating for feature creation.
    ######
    mu <- mean(train_set$rating)
    train_set <- train_set %>%
      mutate(adj = mu)
    
    ######
    # Creating movie-related features.
    ######
    movie_data <- train_set %>% 
      group_by(`Action`) %>%
      mutate(b_g01 = mean(rating - adj), adj = adj + b_g01) %>%
      group_by(`Adventure`) %>%
      mutate(b_g02 = mean(rating - adj), adj = adj + b_g02) %>%
      group_by(`Animation`) %>%
      mutate(b_g03 = mean(rating - adj), adj = adj + b_g03) %>%
      group_by(`Children`) %>%
      mutate(b_g04 = mean(rating - adj), adj = adj + b_g04) %>%
      group_by(`Comedy`) %>%
      mutate(b_g05 = mean(rating - adj), adj = adj + b_g05) %>%
      group_by(`Crime`) %>%
      mutate(b_g06 = mean(rating - adj), adj = adj + b_g06) %>%
      group_by(`Documentary`) %>%
      mutate(b_g07 = mean(rating - adj), adj = adj + b_g07) %>%
      group_by(`Drama`) %>%
      mutate(b_g08 = mean(rating - adj), adj = adj + b_g08) %>%
      group_by(`Fantasy`) %>%
      mutate(b_g09 = mean(rating - adj), adj = adj + b_g09) %>%
      group_by(`Film-Noir`) %>%
      mutate(b_g10 = mean(rating - adj), adj = adj + b_g10) %>%
      group_by(`Horror`) %>%
      mutate(b_g11 = mean(rating - adj), adj = adj + b_g11) %>%
      group_by(`Musical`) %>%
      mutate(b_g12 = mean(rating - adj), adj = adj + b_g12) %>%
      group_by(`Mystery`) %>%
      mutate(b_g13 = mean(rating - adj), adj = adj + b_g13) %>%
      group_by(`Romance`) %>%
      mutate(b_g14 = mean(rating - adj), adj = adj + b_g14) %>%
      group_by(`Sci-Fi`) %>%
      mutate(b_g15 = mean(rating - adj), adj = adj + b_g15) %>%
      group_by(`Thriller`) %>%
      mutate(b_g16 = mean(rating - adj), adj = adj + b_g16) %>%
      group_by(`War`) %>%
      mutate(b_g17 = mean(rating - adj), adj = adj + b_g17) %>%
      group_by(`Western`) %>%
      mutate(b_g18 = mean(rating - adj), adj = adj + b_g18) %>%
      ungroup() %>%
      mutate(b_g = select(., b_g01:b_g18) %>% rowSums()) %>%
      group_by(movieId) %>%
      summarize(movie_ratings = n(),
                b_m = sum(rating - adj)/(n()+l_m) + mean(b_g))
    train_set <- train_set %>%
      left_join(movie_data, by = "movieId") %>%
      mutate(adj = adj + b_m)
    
    ######
    # Creating user-related features
    ######
    user_data <- train_set %>%
      group_by(userId) %>%
      summarize(user_ratings = n(), 
                b_u = sum(rating - adj)/(n()+l_u))
    train_set <- train_set %>%
      left_join(user_data, by = "userId") %>%
      mutate(adj = adj + b_u)
    
    ######
    # Training xgb model.
    ######
    x_train <- train_set %>% 
      select(release_year, review_year, adj,
             movie_ratings, b_m, 
             user_ratings, b_u,
             all_of(genrelist)) %>%
      sapply(as.numeric)
    y_train <- train_set %>%
      pull(rating)
    
    train_xgb <- caret::train(x_train, y_train, 
                              method = "xgbTree",
                              tuneGrid = grid,
                              trControl = control)
    
    ######
    # Generating predictions and calculating RMSE.
    ######
    print("Generating predictions.")
    test_set <- test_set %>%
      mutate(adj = mu) %>%
      left_join(movie_data, by = "movieId") %>%
      left_join(user_data, by = "userId") %>%
      mutate(adj = adj + b_m + b_u)
    
    x_test <- test_set %>% 
      select(release_year, review_year, adj,
             movie_ratings, b_m, 
             user_ratings, b_u,
             all_of(genrelist)) %>%
      sapply(as.numeric)
    y_test <- test_set %>%
      pull(rating)
    
    y_pred <- predict(train_xgb, x_test)
    score <- RMSE(y_pred, y_test) 
    
    print("Done.")
    return(score)
  })
}

# Lambda values to use if skipping running regularization.
basic_lambdas <- list(best_m = 4.3, best_u = 5.3)
g_lambdas <- list(best_m = 4.9, best_u = 5.3)
g_all_lambdas <- list(best_m = 5.4, best_u = 5.3)

# Scoring method for choosing the best lambda.
lambda_score <- function(l){
  2*sum(l) + max(l) - min(l)
}

# Generating user/movie lambdas for a rmse algorithm.
generate_lambdas <- function(FUN){
  print("Getting movie lambda ballpark.")
  lambda_m <- 0
  score <- FUN(l_m = lambda_m) %>% 
    lambda_score()
  search <- T
  safety <- 0
  while(search & safety < 10){
    safety <- safety + 1
    print(paste("Checking l_m =", lambda_m + 1))
    new_score <- FUN(l_m = lambda_m + 1) %>% 
      lambda_score()
    if(new_score < score){
      lambda_m <- lambda_m + 1
      score <- new_score
    } else {
      search <- F
    }
  }
  if(search){
    print("Regularization stopped early.")
  }
  print("Estimating movie lambda.")
  search <- T
  safety <- 0
  while(search & safety < 1){
    safety <- safety + 0.1
    print(paste("Checking l_m =", lambda_m + 0.1))
    new_score <- FUN(l_m = lambda_m + 0.1) %>% 
      lambda_score()
    if(new_score < score){
      lambda_m <- lambda_m + 0.1
      score <- new_score
    } else {
      search <- F
    }
  }
  if(search){
    print("Regularization stopped early.")
  }
  if(safety == 0.1){
    search <- T
    safety <- 0
    while(search & safety < 1){
      safety <- safety + 0.1
      print(paste("Checking l_m =", lambda_m - 0.1))
      new_score <- FUN(l_m = lambda_m - 0.1) %>% 
        lambda_score()
      if(new_score < score){
        lambda_m <- lambda_m - 0.1
        score <- new_score
      } else {
        search <- F
      }
    }
  }
  if(search){
    print("Regularization stopped early.")
  }
  print("Getting user lambda ballpark.")
  lambda_u <- 0
  search <- T
  safety <- 0
  while(search & safety < 10){
    safety <- safety + 1
    print(paste("Checking l_u =", lambda_u + 1))
    new_score <- FUN(l_m = lambda_m,
                     l_u = lambda_u + 1) %>% 
      lambda_score()
    if(new_score < score){
      lambda_u <- lambda_u + 1
      score <- new_score
    } else {
      search <- F
    }
  }
  if(search){
    print("Regularization stopped early.")
  }
  print("Estimating user lambda.")
  search <- T
  safety <- 0
  while(search & safety < 1){
    safety <- safety + 0.1
    print(paste("Checking l_u =", lambda_u + 0.1))
    new_score <- FUN(l_m = lambda_m,
                     l_u = lambda_u + 0.1) %>% 
      lambda_score()
    if(new_score < score){
      lambda_u <- lambda_u + 0.1
      score <- new_score
    } else {
      search <- F
    }
  }
  if(search){
    print("Regularization stopped early.")
  }
  if(safety == 0.1){
    search <- T
    safety <- 0
    while(search & safety < 1){
      safety <- safety + 0.1
      print(paste("Checking l_u =", lambda_u - 0.1))
      new_score <- FUN(l_m = lambda_m,
                       l_u = lambda_u - 0.1) %>% 
        lambda_score()
      if(new_score < score){
        lambda_u <- lambda_u - 0.1
        score <- new_score
      } else {
        search <- F
      }
    }
  }
  if(search){
    print("Regularization stopped early.")
  }
  return(list(best_m = lambda_m, best_u = lambda_u))
}

################
# Regularization
################
basic_lambdas <- generate_lambdas(generate_basic_rmses) 
basic_lambdas # 4.3, 5.3

g_lambdas <- generate_lambdas(generate_g_rmses)
g_lambdas # 4.9, 5.3

g_all_lambdas <- generate_lambdas(generate_g_all_rmses) 
g_all_lambdas # 5.4, 5.3

######
# Comparison of model RMSEs with/without regularization.
######
basic_rmses <- generate_basic_rmses()
basic_rmses # 0.8663835 0.8661278 0.8658875

reg_basic_rmses <- generate_basic_rmses(l_m = basic_lambdas$best_m,
                                        l_u = basic_lambdas$best_u)
reg_basic_rmses # 0.8658067 0.8654997 0.8653244 

g_rmses <- generate_g_rmses()
g_rmses # 0.8663835 0.8661278 0.8658875 

reg_g_rmses <- generate_g_rmses(l_m = g_lambdas$best_m, 
                                l_u = g_lambdas$best_u)
reg_g_rmses # 0.8657921 0.8654794 0.8653105 

g_all_rmses <- generate_g_all_rmses()
g_all_rmses # 0.8663835 0.8661278 0.8658875 

reg_g_all_rmses <- generate_g_all_rmses(l_m = g_all_lambdas$best_m,
                                        l_u = g_all_lambdas$best_u)
reg_g_all_rmses # 0.8657755 0.8654509 0.8652945 

basic_tree_rmses <- generate_basic_tree_rmses()
basic_tree_rmses # 0.8579424 0.8576246 0.8575241 

reg_basic_tree_rmses <- generate_basic_tree_rmses(l_m = basic_lambdas$best_m, 
                                                  l_u = basic_lambdas$best_u)
reg_basic_tree_rmses # 0.8579790 0.8576607 0.8574868 

g_tree_rmses <- generate_g_tree_rmses()
g_tree_rmses # 0.8576938 0.8575256 0.8574749 

reg_g_tree_rmses <- generate_g_tree_rmses(l_m = g_lambdas$best_m,
                                          l_u = g_lambdas$best_u)
reg_g_tree_rmses # 0.8576554 0.8573988 0.8574412 

g_all_tree_rmses <- generate_g_all_tree_rmses()  
g_all_tree_rmses # 0.8573896 0.8571500 0.8569527

reg_g_all_tree_rmses <- generate_g_all_tree_rmses(l_m = g_all_lambdas$best_m,
                                                  l_u = g_all_lambdas$best_u)
reg_g_all_tree_rmses # 0.8573745 0.8572931 0.8571587 

# Saving all RMSE/lambda data.

rmses <- rbind(basic_rmses, g_rmses, g_all_rmses,
               basic_tree_rmses, g_tree_rmses, g_all_tree_rmses,
               reg_basic_rmses, reg_g_rmses, reg_g_all_rmses, 
               reg_basic_tree_rmses, reg_g_tree_rmses, reg_g_all_tree_rmses) %>%
  cbind("Mean" = apply(., 2, mean), "Lambda Score" = apply(., 2, lambda_score))

save(rmses, file = "./Data/Exploration.rda")
rm(rmses,
   basic_rmses, g_rmses, g_all_rmses,
   basic_tree_rmses, g_tree_rmses, g_all_tree_rmses,
   reg_basic_rmses, reg_g_rmses, reg_g_all_rmses, 
   reg_basic_tree_rmses, reg_g_tree_rmses, reg_g_all_tree_rmses)
#######################
# End of Regularization
#######################
######
# Final RMSE Calculation
######
# Calculating final RMSE with full edx-based model on validation set.
prep_validation <- prepare_data(validation)
final_setpair <- list(list(train = prep_edx, 
                           test = prep_validation))

final_rmse <- generate_g_all_tree_rmses(setpair = final_setpair,
                                        l_m = g_all_lambdas$best_m, 
                                        l_u = g_all_lambdas$best_u)
print(paste("Final RMSE:", final_rmse)) # 0.8562876
######
# Removing quickly reproducible but large data and saving the rest.
######
rm(sets, final_setpair)
lambdas <- rbind(basic_lambdas, g_lambdas, g_all_lambdas)
save.image(file = "./Data/Capstone.rda")
Sys.time()

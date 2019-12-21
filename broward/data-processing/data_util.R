#Bhrij Patel
compute_features = function(person_id,screening_date,first_offense_date,current_offense_date,
                            arrest,charge,jail,prison,prob,people) {
  ### Computes features (e.g., number of priors) for each person_id/screening_date.
  
  # pmap coerces dates to numbers so convert back to date.
  first_offense_date = as_date(first_offense_date)
  screening_date = as_date(screening_date)
  current_offense_date = as_date(current_offense_date) 
  
  out = list()
  
  ### ID information
  out$person_id = person_id
  out$screening_date = screening_date
  
  ### Other features
  
  # Number of felonies
  out$p_felony_count_person = ifelse(is.null(charge), 0, sum(charge$is_felony, na.rm = TRUE))
  
  # Number of misdemeanors
  out$p_misdem_count_person  = ifelse(is.null(charge), 0, sum(charge$is_misdem, na.rm = TRUE))
  
  # Number of violent charges
  out$p_violence  = ifelse(is.null(charge), 0, sum(charge$is_violent, na.rm = TRUE))
  
  #p_current_age: Age at screening date
  out$age_at_current_charge = floor(as.numeric(as.period(interval(people$dob,screening_date)), "years"))
  
  #p_age_first_offense: Age at first offense 
  out$age_at_first_charge = floor(as.numeric(as.period(interval(people$dob,first_offense_date)), "years"))
  
  ### History of Violence
  
  #p_juv_fel_count
  #out$p_juv_fel_count = ifelse(is.null(people), 0, people$juv_fel_count)
  out$p_juv_fel_count = ifelse(is.null(charge), 0, sum(charge$is_felony & charge$is_juv,na.rm=TRUE))
  
  #p_felprop_violarrest
  out$p_felprop_viol = ifelse(is.null(charge), 0,sum(charge$is_felprop_violarrest, na.rm = TRUE))
  
  #p_murder_arrest
  out$p_murder = ifelse(is.null(charge), 0, sum(charge$is_murder, na.rm = TRUE))
  
  #p_felassault_arrest
  out$p_felassault = ifelse(is.null(charge), 0, sum(charge$is_felassault_arrest, na.rm = TRUE))
  
  #p_misdemassault_arrest
  out$p_misdeassault = ifelse(is.null(charge), 0, sum(charge$is_misdemassault_arrest, na.rm = TRUE))
  
  #p_famviol_arrest
  out$p_famviol = ifelse(is.null(charge), 0, sum(charge$is_family_violence, na.rm = TRUE))
  
  #p_sex_arrest
  out$p_sex_offense = ifelse(is.null(charge), 0, sum(charge$is_sex_offense, na.rm = TRUE))
  
  #p_weapons_arrest
  out$p_weapon =  ifelse(is.null(charge), 0, sum(charge$is_weapons, na.rm = TRUE))
  
  ### History of Non-Compliance
  
  # Number of offenses while on probation
  out$p_n_on_probation = ifelse(is.null(charge) | is.null(prob), 0, count_on_probation(charge,prob))
  
  # Whether or not current offense was while on probation (two ways)
  if(is.null(prob)){
    out$p_current_on_probation = 0
  } else if(is.na(current_offense_date)) {
    out$p_current_on_probation = NA
  } else {
    out$p_current_on_probation = if_else(count_on_probation(data.frame(offense_date=current_offense_date),prob)>0,1,0)
  }
  
  # Number of times provation was violated or revoked
  out$p_prob_revoke =  ifelse(is.null(prob), 0, sum(prob$is_revoke==1 & prob$EventDate < current_offense_date))
  
  ### Criminal Involvement
  
  # Number of charges / arrests
  out$p_charges = ifelse(is.null(charge), 0, nrow(charge))
  out$p_arrest = ifelse(is.null(arrest), 0, nrow(arrest))
  
  # Number of times sentenced to jail/prison 30 days or more
  out$p_jail30 = ifelse(is.null(jail), 0, sum(jail$sentence_days >= 30, na.rm=TRUE))
  out$p_prison30 = ifelse(is.null(prison), 0, sum(prison$sentence_days >= 30, na.rm=TRUE))
  
  ## added on 10/13/2019
  out$p_incarceration = ifelse(is.null(prison) & is.null(jail), 0, 1)
  
  # Number of prison sentences
  out$p_prison =  ifelse(is.null(prison), 0, nrow(prison))
  
  # Number of times on probation
  out$p_probation =  ifelse(is.null(prob), 0, sum(prob$prob_event== "On" & prob$EventDate < current_offense_date, na.rm = TRUE))
  
  ### Additional features
  
  #Property  charge
  out$p_property =  ifelse(is.null(charge), 0, sum(charge$is_property, na.rm = TRUE))
   
  #Traffic  charges
  out$p_traffic =  ifelse(is.null(charge), 0, sum(charge$is_traffic, na.rm = TRUE))
  
  #Drug charges-have to look into municipalities
  out$p_drug =  ifelse(is.null(charge), 0, sum(charge$is_drug, na.rm = TRUE))

  #DUI charges
  out$p_dui =  ifelse(is.null(charge), 0, sum(charge$is_dui, na.rm = TRUE))

  #Domestic Violence charges
  out$p_domestic =  ifelse(is.null(charge), 0, sum(charge$is_domestic_viol, na.rm = TRUE))

  #Stalking charges
  out$p_stalking =  ifelse(is.null(charge), 0, sum(charge$is_stalking, na.rm = TRUE))

  #Voyeurism charges
  out$p_voyeurism =  ifelse(is.null(charge), 0, sum(charge$is_voyeurism, na.rm = TRUE))

  #Fraud charges
  out$p_fraud =  ifelse(is.null(charge), 0, sum(charge$is_stalking, na.rm = TRUE))

  #Stealing charges
  out$p_stealing =  ifelse(is.null(charge), 0, sum(charge$is_stealing, na.rm = TRUE))

  #Trespass charges
  out$p_trespass =  ifelse(is.null(charge), 0, sum(charge$is_trespass, na.rm = TRUE))

  return(out)
}

compute_features_on = function(person_id,screening_date,first_offense_date,current_offense_date,
                               arrest,charge,jail,prison,prob,people) {
  ### Computes features related to current offense
  
  # pmap coerces dates to numbers so convert back to date.
  first_offense_date = as_date(first_offense_date)
  screening_date = as_date(screening_date)
  current_offense_date = as_date(current_offense_date) 
  
  out = list()
  
  ### ID information
  out$person_id = person_id
  out$screening_date = screening_date
  
  out$is_misdem = ifelse(is.null(charge), NA, if_else(any(charge$is_misdem==1) & all(charge$is_felony==0),1,0))
  
  return(out)
}





compute_outcomes = function(person_id,screening_date,first_offense_date,current_offense_date,
                            arrest,charge,jail,prison,prob,people){
  
  out = list()
  
  # pmap coerces dates to numbers so convert back to date.
  first_offense_date = as_date(first_offense_date)
  screening_date = as_date(screening_date)
  current_offense_date = as_date(current_offense_date)
  
  ### ID information
  out$person_id = person_id
  out$screening_date = screening_date
  
  if(is.null(charge)) {
    #out$nullcharge = 1
    out$general_two_year = 0
    out$general_six_month = 0
    out$years = 0
    out$recidnot = 0
    
    out$drug_two_year = 0
    out$property_two_year = 0
    #out$stalking_two_year = 0
    #out$trespass_two_year = 0
    #out$traffic_two_year = 0
    #out$voyeurism_two_year = 0
    #out$fraud_two_year = 0
    #out$stealing2 = 0
    #out$dui2 = 0
    #out$domestic2 = 0
    #out$murder2 = 0
    out$misdemeanor_two_year = 0
    out$felony_two_year = 0

    out$drug_six_month = 0
    out$property_six_month = 0
    #out$stalking6 = 0
    #out$trespass6 = 0
    #out$traffic6 = 0
    #out$voyeurism6 = 0
    #out$fraud6 = 0
    #out$stealing6 = 0
    #out$dui6 = 0
    #out$domestic6 = 0
    #out$murder6 = 0
    out$misdemeanor_six_month = 0
    out$felony_six_month = 0
    
    out$violent_two_year = 0
    #out$domestic_violent2 = 0
    #out$drug_violent2 = 0
    #out$property_violent2 = 0
    #out$stalking_violent2 = 0
    #out$trespass_violent2 = 0
    #out$traffic_violent2 = 0
    #out$voyeurism_violent2 = 0
    #out$fraud_violent2 = 0
    #out$stealing_violent2 = 0
    #out$dui_violent2 = 0
    #out$murder_violent2 = 0  
    
    out$violent_six_month = 0
    #out$recid_domestic_violent6 = 0
    #out$recid_drug_violent6 = 0
    #out$recid_property_violent6 = 0
    #out$recid_stalking_violent6 = 0
    #out$recid_trespass_violent6 = 0
    #out$recid_traffic_violent6 = 0
    #out$recid_voyeurism_violent6 = 0
    #out$recid_fraud_violent6 = 0
    #out$recid_stealing_violent6 = 0
    #out$recid_dui_violent6 = 0
    #out$recid_murder_violent6 = 0  
    
  } else {
    out$nullcharge = 0
    # Sort charges in ascending order
    charge = charge %>% dplyr::arrange(offense_date)
    # General recidivism
    date_next_offense = charge$offense_date[1]
    years_next_offense = as.numeric(as.period(interval(screening_date,date_next_offense)), "years")
    years_next_offense[is.na(years_next_offense)] = 0
    out$years = years_next_offense

    out$general_two_year = if_else(years_next_offense <= 2 & years_next_offense > 0, 1, 0) 
    out$general_six_month = if_else(years_next_offense <= 0.5 & years_next_offense > 0, 1, 0)
    out$recidnot = as.numeric(!out$general_two_year)
    if(is.na(out$recidnot)){
      out$recidnot = 0
    }

    out$drug_two_year = if_else(out$general_two_year == 1 && charge$is_drug, 1, 0)
    out$property_two_year = if_else(out$general_two_year == 1 && charge$is_property, 1, 0)
    #out$recid_stalking2 = if_else(out$recid_two_year == 1 && charge$is_stalking, 1, 0)
    #out$recid_trespass2 = if_else(out$recid_two_year == 1 && charge$is_trespass, 1, 0)
    #out$recid_traffic2 =  if_else(out$recid_two_year == 1 && charge$is_traffic, 1, 0)
    #out$recid_voyeurism2 =  if_else(out$recid_two_year == 1 && charge$is_voyeurism, 1, 0)
    #out$recid_fraud2 = if_else(out$recid_two_year == 1 && charge$is_fraud, 1, 0)
    #out$recid_stealing2 = if_else(out$recid_two_year == 1 && charge$is_stealing, 1, 0)
    #out$recid_dui2 = if_else(out$recid_two_year == 1 && charge$is_dui, 1, 0)   
    #out$recid_domestic2 = if_else(out$recid_two_year == 1 && charge$is_domestic_viol, 1, 0)
    #out$recid_murder2 = if_else(out$recid_two_year == 1 && charge$is_murder, 1, 0)
    out$misdemeanor_two_year = if_else(out$general_two_year == 1 && charge$is_misdem, 1, 0)
    out$felony_two_year = if_else(out$general_two_year == 1 && charge$is_felony, 1, 0)

    out$drug_six_month = if_else(out$general_six_month == 1 && charge$is_drug, 1, 0)
    out$property_six_month = if_else(out$general_six_month == 1 && charge$is_property, 1, 0)
    #out$recid_stalking6 = if_else(out$recid_six_month == 1 && charge$is_stalking, 1, 0)
    #out$recid_trespass6 = if_else(out$recid_six_month == 1 && charge$is_trespass, 1, 0)
    #out$recid_traffic6 =  if_else(out$recid_six_month == 1 && charge$is_traffic, 1, 0)
    #out$recid_voyeurism6 =  if_else(out$recid_six_month == 1 && charge$is_voyeurism, 1, 0)
    #out$recid_fraud6 = if_else(out$recid_six_month == 1 && charge$is_fraud, 1, 0)
    #out$recid_stealing6 = if_else(out$recid_six_month == 1 && charge$is_stealing, 1, 0)
    #out$recid_dui6 = if_else(out$recid_six_month == 1 && charge$is_dui, 1, 0)   
    #out$recid_domestic6 = if_else(out$recid_six_month == 1 && charge$is_domestic_viol, 1, 0)
    #out$recid_murder6 = if_else(out$recid_six_month == 1 && charge$is_murder, 1, 0)
    out$misdemeanor_six_month = if_else(out$general_six_month == 1 && charge$is_misdem, 1, 0)
    out$felony_six_month = if_else(out$general_six_month == 1 && charge$is_felony, 1, 0)
    
    # Violent recidivism
    date_next_offense_violent = filter(charge,is_violent==1)$offense_date[1]
    
    if(is.na(date_next_offense_violent)) {
      out$violent_two_year = 0
      #out$recid_domestic_violent2 = 0
      #out$recid_drug_violent2 = 0
      #out$recid_property_violent2 = 0
      #out$recid_stalking_violent2 = 0
      #out$recid_trespass_violent2 = 0
      #out$recid_traffic_violent2 = 0
      #out$recid_voyeurism_violent2 = 0
      #out$recid_fraud_violent2 = 0
      #out$recid_stealing_violent2 = 0
      #out$recid_dui_violent2 = 0    
      #out$recid_murder_violent2 = 0  
      
      out$violent_six_month = 0
      #out$recid_domestic_violent6 = 0
      #out$recid_drug_violent6 = 0
      #out$recid_property_violent6 = 0
      #out$recid_stalking_violent6 = 0
      #out$recid_trespass_violent6 = 0
      #out$recid_traffic_violent6 = 0
      #out$recid_voyeurism_violent6 = 0
      #out$recid_fraud_violent6 = 0
      #out$recid_stealing_violent6 = 0
      #out$recid_dui_violent6 = 0    
      #out$recid_murder_violent6 = 0  
      
      } else {
      years_next_offense_violent = as.numeric(as.period(interval(screening_date,date_next_offense_violent)), "years")
      
      out$violent_two_year = if_else(years_next_offense_violent <= 2, 1, 0)
      #out$recid_domestic_violent2 = if_else(years_next_offense_violent <= 2 && charge$is_domestic_viol, 1, 0)   
      #out$recid_drug_violent2 = if_else(years_next_offense_violent <= 2 && charge$is_drug, 1, 0)
      #out$recid_property_violent2 = if_else(years_next_offense_violent <= 2 && charge$is_property, 1, 0)
      #out$recid_stalking_violent2 = if_else(years_next_offense_violent <= 2 && charge$is_stalking, 1, 0)
      #out$recid_trespass_violent2 = if_else(years_next_offense_violent <= 2 && charge$is_trespass, 1, 0)
      #out$recid_traffic_violent2 =  if_else(years_next_offense_violent <= 2 && charge$is_traffic, 1, 0)
      #out$recid_voyeurism_violent2 =  if_else(years_next_offense_violent <= 2 && charge$is_voyeurism, 1, 0)
      #out$recid_fraud_violent2 = if_else(years_next_offense_violent <= 2 && charge$is_fraud, 1, 0)
      #out$recid_stealing_violent2 = if_else(years_next_offense_violent <= 2 && charge$is_stealing, 1, 0)
      #out$recid_dui_violent2 = if_else(years_next_offense_violent <= 2 && charge$is_dui, 1, 0)   
      #out$recid_murder_violent2 = if_else(years_next_offense_violent <= 2 && charge$is_murder, 1, 0)    
      
      out$violent_six_month = if_else(years_next_offense_violent <= 0.5, 1, 0)
      #out$recid_domestic_violent6 = if_else(years_next_offense_violent <= 0.5 && charge$is_domestic_viol, 1, 0)   
      #out$recid_drug_violent6 = if_else(years_next_offense_violent <= 0.5 && charge$is_drug, 1, 0)
      #out$recid_property_violent6 = if_else(years_next_offense_violent <= 0.5 && charge$is_property, 1, 0)
      #out$recid_stalking_violent6 = if_else(years_next_offense_violent <= 0.5 && charge$is_stalking, 1, 0)
      #out$recid_trespass_violent6 = if_else(years_next_offense_violent <= 0.5 && charge$is_trespass, 1, 0)
      #out$recid_traffic_violent6 =  if_else(years_next_offense_violent <= 0.5 && charge$is_traffic, 1, 0)
      #out$recid_voyeurism_violent6 =  if_else(years_next_offense_violent <= 0.5 && charge$is_voyeurism, 1, 0)
      #out$recid_fraud_violent6 = if_else(years_next_offense_violent <= 0.5 && charge$is_fraud, 1, 0)
      #out$recid_stealing_violent6 = if_else(years_next_offense_violent <= 0.5 && charge$is_stealing, 1, 0)
      #out$recid_dui_violent6 = if_else(years_next_offense_violent <= 0.5 && charge$is_dui, 1, 0)   
      #out$recid_murder_violent6 = if_else(years_next_offense_violent <= 0.5 && charge$is_murder, 1, 0)    
      }
  }
  
  return(out)
  
  
}



compute_past_crimes = function(person_id,screening_date,first_offense_date,current_offense_date,
                            arrest,charge,jail,prison,prob,people){
    
    out = list()
    
    # pmap coerces dates to numbers so convert back to date.
    first_offense_date = as_date(first_offense_date)
    screening_date = as_date(screening_date)
    current_offense_date = as_date(current_offense_date)
    
    ### ID information
    out$person_id = person_id
    out$screening_date = screening_date
    
    if(is.null(charge)) {
        out$years_since_last_crime = 0
        out$six_month = 0
        out$one_year = 0
        out$three_year = 0
        out$five_year = 0
    } else {
        year_offenses = as.numeric(as.period(interval(charge$offense_date, screening_date)), "years")
        out$years_since_last_crime = min(year_offenses)
        
        if (any(year_offenses <= 0.5)) {
            out$six_month = 1
            out$one_year = 1
            out$three_year = 1
            out$five_year = 1
        } else if (any(year_offenses <= 1) & (all(year_offenses > 0.5))) {
            out$six_month = 0
            out$one_year = 1
            out$three_year = 1
            out$five_year = 1
        } else if (any(year_offenses <= 3) & (all(year_offenses > 1))) {
            out$six_month = 0
            out$one_year = 0
            out$three_year = 1
            out$five_year = 1
        } else if (any(year_offenses <= 5 & all(year_offenses > 3))){
            out$six_month = 0
            out$one_year = 0
            out$three_year = 0
            out$five_year = 1
        } else if (all(year_offenses > 5)) {
            out$six_month = 0
            out$one_year = 0
            out$three_year = 0
            out$five_year = 0
        }
    }
    
    return(out)
}



#compute_outcomes_graph = function(person_id,screening_date,first_offense_date,current_offense_date,
#                            arrest,charge,jail,prison,prob,people){
  
#  out = list()
  
  # pmap coerces dates to numbers so convert back to date.
#  first_offense_date = as_date(first_offense_date)
#  screening_date = as_date(screening_date)
#  current_offense_date = as_date(current_offense_date)
  
  ### ID information
#  out$person_id = person_id
#  out$screening_date = screening_date
  
#  drug_amt = 0
#  property_amt = 0
#  stalking_amt = 0
#  dom_amt = 0
  
#  if(is.null(charge)) {
#    out$recid = 0
#    out$recid_drug = 0
#    out$recid_property = 0
#    out$recid_stalking = 0
    
#    out$recid_violent = 0
#    out$recid_domestic_violent = 0
    
#  } else {
    
    # Sort charges in ascending order
#    charge = charge %>% dplyr::arrange(offense_date)
    
    # General recidivism
#    date_next_offense = charge$offense_date[1]
#    years_next_offense = as.numeric(as.period(interval(screening_date,date_next_offense)), "years")
#    out$recid = if_else(years_next_offense<= 2, 1, 0)
    
#    out$recid_drug = if_else(years_next_offense <= 2 && charge$is_drug, 1, 0)
    
#    drug_amt = if_else(years_next_offense <= 2 && charge$is_drug, drug_amt+1, drug_amt)
    
#    out$recid_property = if_else(years_next_offense <= 2 && charge$is_property, 1, 0)
    
#    property_amt = if_else(years_next_offense <= 2 && charge$is_property, property_amt+1, property_amt)
    
#    out$recid_stalking = if_else(years_next_offense<= 2 && charge$is_stalking, 1, 0)
    
#    stalking_amt = if_else(years_next_offense <= 2 && charge$is_stalking, stalking_amt+1, stalking_amt)
    
    # Violent recidivism
#    date_next_offense_violent = filter(charge,is_violent==1)$offense_date[1]
    
#    if(is.na(date_next_offense_violent)) {
#      out$recid_violent = 0
#      out$recid_domestic_violent = 0
      
#    } else {
#      years_next_offense_violent = as.numeric(as.period(interval(screening_date,date_next_offense_violent)), "years")
#      out$recid_violent = if_else(years_next_offense_violent <= 2, 1, 0)
      
#      out$recid_domestic_violent = if_else(years_next_offense_violent <= 2 && charge$is_domestic_viol, 1, 0)     
      
#      dom_amt = if_else(years_next_offense <= 2 && charge$is_domestic_viol, dom_amt+1, dom_amt)
      
#    }
#  }
  
#  crime <- c("Drug", "Property", "Stalking", "Domestic Violence")
  
#  amount <- c(drug_amt, property_amt, stalking_amt, dom_amt) 
#  graph_df <- data.frame(crime, amount)
  
#  return(graph_df)
  
  
#}

count_on_probation = function(charge, prob){
  
  # Make sure prob is sorted in ascending order of EventDate
  
  u_charge = charge %>%
    group_by(offense_date) %>%
    summarize(count = n()) %>%
    mutate(rank = findInterval(as.numeric(offense_date), as.numeric(prob$EventDate)))  %>%
    group_by(rank) %>%
    mutate(
      event_before = ifelse(rank==0, NA, prob$prob_event[rank]),
      days_before = ifelse(rank==0, NA, floor(as.numeric(as.period(interval(prob$EventDate[rank],offense_date)), "days"))),
      event_after = ifelse(rank==nrow(prob), NA, prob$prob_event[rank+1]),
      days_after = ifelse(rank==nrow(prob),NA, floor(as.numeric(as.period(interval(offense_date, prob$EventDate[rank+1])), "days")))
    ) %>%
    mutate(is_on_probation = pmap(list(event_before, days_before, event_after, days_after), .f=classify_charge)) %>%
    unnest()
  
  return(sum(u_charge$count[u_charge$is_on_probation]))
}



classify_charge = function(event_before, days_before, event_after, days_after,
                           thresh_days_before=365, thresh_days_after=30) {
  
  if (is.na(event_before)) {
    # No events before
    if (event_after == "Off" & days_after <= thresh_days_after) {
      return(TRUE)
    }
    
  } else if (is.na(event_after)) {
    # No events after
    if (event_before == "On" & days_before <= thresh_days_before) {
      return(TRUE)
    }
  }
  
  else { # Neither event is NA
    
    if (event_before=="On" & event_after=="Off") {
      return(TRUE)
      
    } else if (event_before=="On" & days_before <= thresh_days_before & event_after=="On") {
      return(TRUE)
      
    } else if (event_before=="Off" & event_after=="Off" & days_after <= thresh_days_after) {
      return(TRUE)
    } 
  }
  return(FALSE)
}


rmse = function(y, yhat) {
  sqrt(mean((y-yhat)^2))
}

# fit_xgboost <- function(train, param) {
#   ###
#   # Cross validates each combination of parameters in param and returns best model
#   # param is a list of xgboost parameters as vectors
#   # train is formatted for xgboost input
#   ###
#   
#   param_df = expand.grid(param) # Each row is a set of parameters to be cross validated
#   n_param = nrow(param_df)
#   
#   ## Allocate space for performance statistics (and set seeds)
#   performance = data.frame(
#     i_param = 1:n_param,
#     seed = sample.int(10000, n_param),
#     matrix(NA,nrow=2,ncol=5,
#            dimnames=list(NULL,
#                          c("iter","train_rmse_mean","train_rmse_std","test_rmse_mean","test_rmse_std"))))
#   col_eval_log = 3:7 # Adjust manually. Column index in performance of evaluation_log output from xgb.cv
#   
#   cat("Training on",n_param,"sets of parameters.\n")
#   
#   ## Loop through the different parameters sets
#   for (i_param in 1:n_param) {
#     
#     set.seed(performance$seed[i_param])
#     
#     mdcv = xgb.cv(data=train, 
#                   params = list(param_df[i_param,])[[1]], 
#                   nthread=6, 
#                   nfold=5, 
#                   nrounds=10000,
#                   verbose = FALSE, 
#                   early_stopping_rounds=50, 
#                   maximize=FALSE)
#     
#     performance[i_param,col_eval_log] = mdcv$evaluation_log[mdcv$best_iteration,]
#   }
#   
#   ## Train on best parameters using best number of rounds
#   i_param_best = performance$i_param[which.min(performance$test_rmse_mean)]
#   print(t(param_df[i_param_best,])) #Prints the best parameters
#   
#   set.seed(performance$seed[i_param_best])
#   
#   mdl_best = xgb.train(data=train, 
#                        params=list(param_df[i_param_best,])[[1]], 
#                        nrounds=performance$iter[i_param_best], 
#                        nthread=6)
#   
#   return(mdl_best)
# }
# 
# 
# fit_svm <- function(formula, train, param) {
#   ###
#   # Cross validates each combination of parameters in param and returns best model
#   # param is a list of svm parameters as vectors
#   # svm parameters are cost, epsilon, and gamma_scale (a scaling factor on the default gamma value)
#   # train is formatted for xgboost input
#   ###
#   
#   
#   param_df = expand.grid(param) # Each row is a set of parameters to be cross validated
#   n_param = nrow(param_df)
#   
#   ## Compute default gamma parameter
#   gamma_default = 1/ncol(train)
#   param_df = param_df %>%
#     mutate(gamma = gamma_scale * gamma_default)
#   
#   ## Make sure only one type parameter
#   if(length(param$type) == 1){
#     
#     if(str_detect(param$type,'regression')){
#       reg_or_class = 'reg'
#     } else if (str_detect(param$type,'classification')) {
#       reg_or_class = 'class'
#     }
#   } else{
#     stop('Can only handle one type parameter')
#   }
#   
#   ## Allocate space for performance statistics (and set seeds)
#   performance = rep(NA,n_param)
#   
#   cat("Training on",n_param,"sets of parameters.\n")
#   
#   ## Loop through the different parameters sets
#   for (i_param in 1:n_param) {
#     
#     mdcv = suppressWarnings(e1071::svm(formula = formula, 
#                                        data = train, 
#                                        type = param$type,
#                                        kernel = 'radial',
#                                        gamma = param_df$gamma[i_param],
#                                        epsilon = param_df$epsilon[i_param],
#                                        cost = param_df$cost[i_param],
#                                        cross = 5,
#                                        scale = TRUE))
#     
#     if(reg_or_class == "reg"){
#       performance[i_param] = mdcv$tot.MSE
#     } else if (reg_or_class == "class") {
#       performance[i_param] = mdcv$tot.accuracy
#     }
#     
#   }
#   
#   ## Train on best parameters using best number of rounds
#   if(reg_or_class == "reg"){
#     i_param_best = which.min(performance)
#   } else if (reg_or_class == "class") {
#     i_param_best = which.max(performance)
#   }
#   
#   print("Best parameters:")
#   print(t(param_df[i_param_best,]))
#   
#   mdl_best = suppressWarnings(e1071::svm(formula = formula, 
#                                          data = train, 
#                                          type = param$type,
#                                          kernel = 'radial',
#                                          gamma = param_df$gamma[i_param_best],
#                                          epsilon = param_df$epsilon[i_param_best],
#                                          cost = param_df$cost[i_param_best],
#                                          cross = 5,
#                                          scale = TRUE))
#   
#   return(mdl_best)
# }
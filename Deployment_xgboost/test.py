from custom_modeler import custom_modeler

user_input = {
    "borough": "Manhattan",
    "bedrooms": 2,
    "bathrooms": 1,
    "size_sqft": 1000,
    "min_to_subway": 5,
    "building_age_yrs": 30,    
    "floor": 5,
    "no_fee": 0,
    "has_roofdeck": 0,
    "has_patio": 0,
    "has_gym": 0,
    "has_washer_dryer": 1,
    "has_doorman": 0,
    "has_elevator": 1,
    "has_dishwasher": 0
}

print(custom_modeler(user_input))

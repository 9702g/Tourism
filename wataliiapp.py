import streamlit as st
import pandas as pd
import pickle
from os.path import dirname, join, realpath

# Header
st.header("Kenya Tourism Expenditure Prediction")
st.subheader("A simple machine learning app to predict how much money a tourist will spend when visiting Kenya.")
st.image("V:\\Kenyan data\\new_tou_app\\images (1).jpg")

# Form
my_form = st.form(key="financial_form")

# Input for country
country = my_form.selectbox("Country", [
    "SWITZERLAND", "UNITED KINGDOM", "CHINA", "SOUTH AFRICA", "UNITED STATES OF AMERICA",
    "NIGERIA", "INDIA", "BRAZIL", "CANADA", "MALTA", "MOZAMBIQUE", "RWANDA", "AUSTRIA",
    "MYANMAR", "GERMANY", "KENYA", "ALGERIA", "IRELAND", "DENMARK", "SPAIN", "FRANCE",
    "ITALY", "EGYPT", "QATAR", "MALAWI", "JAPAN", "SWEDEN", "NETHERLANDS", "UAE", "UGANDA",
    "AUSTRALIA", "YEMEN", "NEW ZEALAND", "BELGIUM", "NORWAY", "ZIMBABWE", "ZAMBIA", "CONGO",
    "BULGARIA", "PAKISTAN", "GREECE", "MAURITIUS", "DRC", "OMAN", "PORTUGAL", "KOREA",
    "SWAZILAND", "TUNISIA", "KUWAIT", "DOMINICA", "ISRAEL", "FINLAND", "CZECH REPUBLIC",
    "UKRAINE", "ETHIOPIA", "BURUNDI", "SCOTLAND", "RUSSIA", "GHANA", "NIGER", "MALAYSIA",
    "COLOMBIA", "LUXEMBOURG", "NEPAL", "POLAND", "SINGAPORE", "LITHUANIA", "HUNGARY",
    "INDONESIA", "TURKEY", "TRINIDAD AND TOBAGO", "IRAQ", "SLOVENIA", "UNITED ARAB EMIRATES",
    "COMORO", "SRI LANKA", "IRAN", "MONTENEGRO", "ANGOLA", "LEBANON", "SLOVAKIA", "ROMANIA",
    "MEXICO", "LATVIA", "CROATIA", "CAPE VERDE", "SUDAN", "COSTA RICA", "CHILE", "NAMIBIA",
    "TAIWAN", "SERBIA", "LESOTHO", "GEORGIA", "PHILIPPINES", "IVORY COAST", "MADAGASCAR",
    "DJIBOUTI", "CYPRUS", "ARGENTINA", "URUGUAY", "MOROCCO", "THAILAND", "BERMUDA", "ESTONIA",
    "BOTSWANA", "VIETNAM", "GUINEA", "MACEDONIA", "HAITI", "LIBERIA", "SAUDI ARABIA", "BOSNIA",
    "BULGARIA", "PERU", "BANGLADESH", "JAMAICA", "SOMALIA"
])

# Input for age group
age_group = my_form.selectbox("Age Group", ["1-24", "25-44", "45-64", "65+"])

# Input for travel with
travel_with = my_form.selectbox("Travel With", ["Friends/Relatives", "Alone", "Spouse", "Children", "Spouse and Children"])

# Input for total number of females
total_female = my_form.number_input("Total Number of Females", min_value=0)

# Input for total number of males
total_male = my_form.number_input("Total Number of Males", min_value=0)

# Input for purpose
purpose = my_form.selectbox("Purpose", [
    "Leisure and Holidays", "Visiting Friends and Relatives", "Business",
    "Meetings and Conference", "Volunteering", "Scientific and Academic", "Other"
])

# Input for main activity
main_activity = my_form.selectbox("Main Activity", [
    "Wildlife tourism", "Cultural tourism", "Mountain climbing", "Beach tourism",
    "Conference tourism", "Hunting tourism", "Bird watching", "Business", "Diving and Sport Fishing"
])

# Input for tour arrangement
tour_arrangement = my_form.selectbox("Tour Arrangement", ["Independent", "Package Tour"])

# Input for package_transport_international
package_transport_international = my_form.selectbox("Package Transport International", ["No", "Yes"])

# Input for package_food
package_food = my_form.selectbox("Package Food", ["No", "Yes"])

# Input for package_transport_local
package_transport_local = my_form.selectbox("Package Transport Local", ["No", "Yes"])

# Input for package_sightseeing
package_sightseeing = my_form.selectbox("Package Sightseeing", ["No", "Yes"])

# Input for package_guided_tour
package_guided_tour = my_form.selectbox("Package Guided Tour", ["No", "Yes"])

# Input for package_insurance
package_insurance = my_form.selectbox("Package Insurance", ["No", "Yes"])

# Input for nights_stayed
nights_stayed = my_form.number_input("Nights Stayed", min_value=0)

# Input for payment_mode
payment_mode = my_form.selectbox("Payment Mode", ["Cash", "Credit Card", "Other", "Travellers Cheque"])

# Input for first_trip
first_trip = my_form.selectbox("First Trip", ["No", "Yes"])

# Input for most_impressing
most_impressing = my_form.text_input("Most Impressions")

# Button to make prediction
if my_form.form_submit_button("Make Prediction"):
    # Prepare input data
    input_data = {
        "country": country, "age_group": age_group, "travel_with": travel_with,
        "total_female": total_female, "total_male": total_male,
        "purpose": purpose, "main_activity": main_activity,
        "tour_arrangement": tour_arrangement,
        "package_transport_international": package_transport_international,
        "package_food": package_food, "package_transport_local": package_transport_local,
        "package_sightseeing": package_sightseeing, "package_guided_tour": package_guided_tour,
        "package_insurance": package_insurance, "nights_stayed": nights_stayed,
        "payment_mode": payment_mode, "first_trip": first_trip,
        "most_impressing": most_impressing
    }
with open(
    join(dirname(realpath(__file__)), "model/xgb_model.pkl"),
    "rb",
) as f:
    model = pickle.load(f)
    # Load the model
    
    else:
        st.error("Model file not found. Please upload a valid model file.")

    # Create DataFrame from input data
    data = pd.DataFrame(input_data, index=[0])

    # Factorize object columns
    for colname in data.select_dtypes("object"):
        data[colname] = data[colname].factorize()[0]

    # Perform prediction
    prediction = model.predict(data)

    # Display results
    st.header("Results")
    st.write("Estimated expenditure: ${:.2f}".format(prediction[0]))

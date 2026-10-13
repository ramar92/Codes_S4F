import pandas as pd 

from dotenv import load_dotenv 

 

from langchain_core.prompts import ChatPromptTemplate 

from langchain_groq import ChatGroq 

 

# Load environment variables 

load_dotenv() 

 

# Read dataset 

df = pd.read_csv("carbon_data.csv") 

 

# Select one vehicle record 

vehicle = df.iloc[0] 

 

# Create prompt template 

prompt = ChatPromptTemplate.from_template(""" 

You are an Automotive Environmental Sustainability Consultant. 

 

Analyze the following vehicle's fuel consumption and carbon emissions. 

 

Vehicle Details 

 

Model Year: {year} 

 

Manufacturer: {make} 

 

Model: {model} 

 

Vehicle Class: {vehicle_class} 

 

Engine Size: {engine_size} L 

 

Cylinders: {cylinders} 

 

Transmission: {transmission} 

 

Fuel Type: {fuel_type} 

 

Fuel Consumption (City): {city} L/100 km 

 

Fuel Consumption (Highway): {highway} L/100 km 

 

Combined Fuel Consumption: {combined} L/100 km 

 

Fuel Economy: {mpg} MPG 

 

CO₂ Emissions: {co2} g/km 

 

Generate a detailed report containing: 

 

1. Overall Environmental Impact 

2. Fuel Efficiency Analysis 

3. CO₂ Emission Analysis 

4. Whether this vehicle is Environment Friendly (Yes/No with reason) 

5. Three Recommendations to Reduce Carbon Emissions 

6. Eco-Friendliness Score (1-10) 

7. Final Conclusion 

 

Present the report in a professional and easy-to-read format. 

""") 

 

# Initialize Groq LLM 

llm = ChatGroq( 

    model="llama-3.3-70b-versatile", 

    temperature=0.3 

) 

 

# Build workflow 

chain = prompt | llm 

 

# Invoke the chain 

response = chain.invoke({ 

    "year": vehicle["MODELYEAR"], 

    "make": vehicle["MAKE"], 

    "model": vehicle["MODEL"], 

    "vehicle_class": vehicle["VEHICLECLASS"], 

    "engine_size": vehicle["ENGINESIZE"], 

    "cylinders": vehicle["CYLINDERS"], 

    "transmission": vehicle["TRANSMISSION"], 

    "fuel_type": vehicle["FUELTYPE"], 

    "city": vehicle["FUELCONSUMPTION_CITY"], 

    "highway": vehicle["FUELCONSUMPTION_HWY"], 

    "combined": vehicle["FUELCONSUMPTION_COMB"], 

    "mpg": vehicle["FUELCONSUMPTION_COMB_MPG"], 

    "co2": vehicle["CO2EMISSIONS"] 

}) 

 

# Print report 

print(response.content) 

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load API key
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Agriculture AI Assistant",
    page_icon="🌱",
    layout="centered"
)

# Header
st.title("🌱 Agriculture AI Assistant")
st.write(
    "Your AI assistant for crop selection, soil health, "
    "irrigation, fertilizers, and plant care."
)

st.divider()

# Sidebar
with st.sidebar:
    st.header("🌾 Farm Details")

    location = st.text_input(
        "Location",
        placeholder="e.g., Tamil Nadu, India"
    )

    crop = st.selectbox(
        "Select Crop",
        [
            "Rice",
            "Wheat",
            "Maize",
            "Cotton",
            "Sugarcane",
            "Tomato",
            "Groundnut",
            "Other"
        ]
    )

    soil = st.selectbox(
        "Soil Type",
        [
            "Unknown",
            "Clay",
            "Sandy",
            "Loamy",
            "Black soil",
            "Red soil"
        ]
    )

    growth_stage = st.selectbox(
        "Crop Growth Stage",
        [
            "Planning",
            "Sowing",
            "Vegetative growth",
            "Flowering",
            "Harvesting"
        ]
    )

# Question input
st.subheader("💬 Ask Your Agriculture Question")

question = st.text_area(
    "What would you like to know?",
    placeholder=(
        "Example: My rice plants have yellow leaves. "
        "What could be the reason?"
    ),
    height=120
)

# Model and prompt
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful Agriculture AI Assistant.

        Give practical, simple, farmer-friendly guidance.

        Consider the farmer's location, crop, soil type,
        and growth stage when relevant.

        Structure the response using:
        1. Possible causes or explanation
        2. Recommended actions
        3. Prevention tips
        4. When expert help is needed

        Do not claim certainty when information is insufficient.
        Ask for more details when necessary.

        For pesticides and fertilizers, avoid guessing doses.
        Recommend following the product label and local
        agricultural expert guidance.

        You provide general educational information, not
        a substitute for laboratory soil testing or diagnosis
        by an agricultural specialist.
        """
    ),
    (
        "human",
        """
        Location: {location}
        Crop: {crop}
        Soil type: {soil}
        Growth stage: {growth_stage}

        Farmer's question: {question}
        """
    )
])

# Generate response
if st.button("🌿 Get Agriculture Advice", type="primary"):

    if not question.strip():
        st.warning("Please enter your agriculture question.")

    elif not os.getenv("GROQ_API_KEY"):
        st.error(
            "GROQ_API_KEY is missing. Please check your .env file."
        )

    else:
        try:
            with st.spinner("Analyzing your question..."):

                llm = ChatGroq(
                    model="openai/gpt-oss-120b",
                    api_key=os.getenv("GROQ_API_KEY"),
                    temperature=0.2
                )

                chain = prompt | llm

                response = chain.invoke({
                    "location": location or "Not specified",
                    "crop": crop,
                    "soil": soil,
                    "growth_stage": growth_stage,
                    "question": question
                })

            st.subheader("🌱 Agriculture Recommendations")
            st.markdown(response.content)

        except Exception as e:
            st.error("Could not generate agriculture advice.")
            st.caption(
                "Check your API key, model access, internet, "
                "and installed package versions."
            )
            st.code(str(e))

st.divider()
st.caption(
    "Built with Streamlit + LangChain + Groq | "
    "For educational purposes"
)

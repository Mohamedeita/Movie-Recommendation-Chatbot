
import streamlit as st

from recommender import chatbot_response


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="MovieMind",
    page_icon="🎬",
    layout="centered"
)


# ==========================================
# Title
# ==========================================

st.title("🎬 MovieMind")

st.write(
    "A movie recommendation chatbot "
    "powered by Machine Learning."
)

st.write(
    "Tell me about a movie you liked, "
    "and I will recommend similar movies."
)

st.divider()


# ==========================================
# Session State
# ==========================================

if "messages" not in st.session_state:

    st.session_state.messages = [

        {
            "role": "assistant",

            "content": (
                "Hello! 👋 I'm MovieMind.\n\n"
                "Tell me a movie you liked, "
                "and I'll recommend similar movies 🎬"
            )
        }

    ]


if "last_movie" not in st.session_state:

    st.session_state.last_movie = None


# ==========================================
# Display Chat History
# ==========================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ==========================================
# Chat Input
# ==========================================

user_message = st.chat_input(
    "Type your message..."
)


# ==========================================
# Process User Message
# ==========================================

if user_message:

    # Show user message
    with st.chat_message("user"):

        st.markdown(
            user_message
        )

    # Save user message
    st.session_state.messages.append({

        "role": "user",

        "content": user_message

    })


    # Get Recommendation
    result = chatbot_response(

        message=user_message,

        n=5,

        last_movie=st.session_state.last_movie

    )


    # No Movie Found
    if result["movie"] is None:

        assistant_message = result["message"]

        with st.chat_message("assistant"):

            st.markdown(
                assistant_message
            )

        st.session_state.messages.append({

            "role": "assistant",

            "content": assistant_message

        })


    # Movie Found
    else:

        st.session_state.last_movie = (
            result["movie"]
        )

        with st.chat_message("assistant"):

            st.markdown(
                result["message"]
            )

            st.write("")

            for i, movie in enumerate(
                result["recommendations"],
                1
            ):

                st.markdown(
                    f"**{i}. 🎬 {movie['title']}**"
                )

                st.caption(
                    f"Genres: {movie['genres']} "
                    f"| Similarity: "
                    f"{movie['similarity']}"
                )


        response_text = (
            result["message"] + "\n\n"
        )

        for i, movie in enumerate(
            result["recommendations"],
            1
        ):

            response_text += (
                f"{i}. 🎬 "
                f"{movie['title']}\n"
            )

        st.session_state.messages.append({

            "role": "assistant",

            "content": response_text

        })


# ==========================================
# Sidebar
# ==========================================

with st.sidebar:

    st.header("🎬 MovieMind")

    st.write(
        "Movie recommendation chatbot "
        "using Machine Learning."
    )

    st.divider()

    st.subheader("How does it work?")

    st.write(
        "1. Enter a movie you liked."
    )

    st.write(
        "2. The system identifies the movie."
    )

    st.write(
        "3. TF-IDF converts movie features "
        "into numerical vectors."
    )

    st.write(
        "4. Cosine Similarity compares movies."
    )

    st.write(
        "5. The system recommends the most "
        "similar movies."
    )

    st.divider()

    # ==========================================
    # Developer Info
    # ==========================================

    st.write("**Created by Mohamed Eita**")

    st.markdown(
        "[LinkedIn](https://www.linkedin.com/in/mohamed-eita-581187371)"
    )

    st.caption(
        "MovieMind | Machine Learning Project"
    )


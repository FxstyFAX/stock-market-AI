import bcrypt
import streamlit as st

users = {
    "admin": "$2b$12$Mf2jq179U5L4JFH0vaKFFuAyuRhrzW0i5FxvoTXGRo0C1bg/.g8o2",
    "user1": bcrypt.hashpw("mypassword".encode(), bcrypt.gensalt()).decode(),
}

def login(username, password):
    if username in users:
        hashed = users[username].encode()
        return bcrypt.checkpw(password.encode(), hashed)
    return False

def show_login():
    st.title("🔐 Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if login(username, password):
            st.session_state.logged_in = True
            st.session_state.username = username
            st.experimental_rerun()
        else:
            st.error("Invalid credentials")

def require_login():
    if "logged_in" not in st.session_state or not st.session_state.logged_in:
        show_login()
        st.stop()

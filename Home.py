import streamlit as st
from PIL import Image

def main():  # [ST4] IMAGE, MULTIPAGE, SIDEBAR
    st.set_page_config(page_title = "SkillScope")
    st.title("Welcome to SkillScope!")  # PAGE TITLE
    image = Image.open("/workspaces/skillscope/logo.png")  # OPEN IMAGE
    st.image(image, width=750)  # DISPLAY IMAGE
    st.write("Test")  #DESCRIPTION
    st.sidebar.success("Select a page")  # SIDEBAR INSTRUCTIONS

main()
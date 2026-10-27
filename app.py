import streamlit as st
from rag import answer

st.title("NPPA Retail Price Assistant")
st.caption("Answers come from retrieved rows of an official NPPA retail price list, with sources.")

q = st.text_input("Ask about a medicine's retail price")
if q:
    with st.spinner("Searching..."):
        out = answer(q)
    if out["answer"].startswith("LLM error") or out["answer"].startswith("No answer"):
        st.warning("The demo's free AI quota may be used up. Please try again later. "
                   "Retrieved source rows are still shown below.")
    else:
        st.write(out["answer"])
    st.subheader("Sources")
    for s in out["sources"]:
        with st.expander(f"{s['source']}, page {s['page']}"):
            st.write(s["text"])
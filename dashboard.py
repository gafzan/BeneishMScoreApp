"""dashboard.py"""

import streamlit as st
from edgar import set_identity, Company, XBRL

set_identity('gafzan@gmail.com')


def on_ticker_change():
    """Handle ticker change event"""
    ticker = st.session_state.ticker_input.upper()

    if ticker:
        company = Company(ticker)
        if company.latest_tenq:
            filing = company.get_filings(form=['10-Q']).latest()
            xbrl = XBRL.from_filing(filing)
            df = xbrl.statements.income_statement(skip_concept_check=False).to_dataframe(standard=False)
            st.write(company.latest_tenq)
            st.dataframe(df)
        else:
            st.warning('No 10-Q available')


def main():

    # Title
    st.title("Manipulation-Score 📈")

    # SEC download section
    st.subheader("Download data from SEC for U.S. stocks")
    st.text_input("🔎 Enter ticker", placeholder="🔎 Enter ticker", key="ticker_input", on_change=on_ticker_change,
                  label_visibility='collapsed')


if __name__ == '__main__':
    main()



import streamlit as st


def intro_page(st):
    st.header("Denmark statistics GPT", divider="rainbow")
    st.markdown(
        """Meet your personal GPT research assistant that can access all the publicly available data at Denmarks Statistics😄. Whatever questions you have about Danish society. Ask away!.

A 5 minute introduction and demo of the program can be viewed [here](https://youtu.be/pBPuyM_DMk4). For the best experience see the quick demo first😉.

In the upper left corner you can choose between English and Danish. The quality of the answers is sometimes better in english. Mostly due to out of the box support for english being better. When time allows I'll fix this.

This is a quick prototype developed in my sparetime. So expect weird bugs and unintented behavior to happen from time to time. Also due to rate limits on Denmark Statistics and OpenAI's API then the response time can be slow if a large number of users are using the app at the same time. If you experience this then please try again later😉."""
    )
    st.subheader("Use of command line")
    st.markdown(
        """The main form of interaction is by writing questions/commands. Your can write 3 different forms of requests.

 1) Specific questions such as "how has the unemployment evolved".
 2) Updates or changes to an existing question such as "include both sexes".
 3) Exploratory requests about the tables or information available at Denmark Statistics. An example could be: "How could I go about looking into unemployment".

When asking a specific question then GPT will find an appropriate table and device the right query to send to Denmark Statistics API. GPT will then provide and overview of the table, plot the data and add the possibility to filter the data. In order to limit the demand on Denmark Statistics API then GPT will try to request aggregate numbers for the variables where a question does not imply that this variable is important. As an example the question "How has unemployment evolved" will typically specify to Denmarks Statistics API to get the aggregate number for Men and Women since the distinction sex is not important to the question. If your are interested in differences depending on sex then you can either ask the question "How has unemployment evolved over time and for different sexes" or simply ask GPT to update the existing request by writing "include both sexes".

For more control of, which tables GPT should use then ask a more exploratory question and then GPT will give you an overview of the different tables available in Denmark Statistics. You can then select specific tables or categories that you wanna limit your search to. Once you have clicked the button "Use selected tables" then all subsequent question will only consider the selected tables. That is until you click the button "removed table selection"."""
    )
    st.subheader("Layout")
    st.markdown(
        """The sidebar is where the GPT messages are streamed and whenever a question is asked then the center of the app is where the plots will appear. To the right will be filter menus for filtering the data retreived from the Denmark Statistics API, a table overview and an expandable box showing similar available tables."""
    )
    st.subheader("Closing remarks")
    st.markdown(
        """This is a quick demo of how one can connect GPT with Denmarks Statistics excellent [public API](https://www.dst.dk/da/Statistik/brug-statistikken/muligheder-i-statistikbanken/api).

Everything about this demo can be made better, faster and cheaper.

And remember this is the worst this technology will ever be. And it is only just getting started😮."""
    )
    st.subheader("Contact")
    st.markdown("Feel free to contact me at jonathanscharff@gmail.com")

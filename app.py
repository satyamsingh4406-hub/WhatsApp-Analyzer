import streamlit as st
import preprocessor, helper
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
nltk.download('vader_lexicon')

from helper import most_common_words

st.sidebar.title('Whatsapp Chat Analyzer')

uploaded_file = st.sidebar.file_uploader("Choose a file")
if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data=bytes_data.decode('utf-8')
    df=preprocessor.preprocess(data)

    #fetch unique user
    user_list=df['user'].unique().tolist()
    user_list.remove('group_notification')
    user_list.sort()
    user_list.insert(0,'Overall')

    selected_user =st.sidebar.selectbox("Show analysis wrt",user_list)

    if st.sidebar.button("Show Analysis"):

        #stats Area
        num_messages, words, num_media_messages,num_links=helper.fetch_stats(selected_user,df)
        st.title("Top Statistics")
        col1, col2, col3, col4= st.columns(4)

        with col1:
            st.header("Total Messages")
            st.title(num_messages)
        with col2:
            st.header("Total Words")
            st.title(words)
        with col3:
            st.header("Media Shared")
            st.title(num_media_messages)
        with col4:
            st.header("Links Shared")
            st.title(num_links)

        #Monthly Timeline
        st.title("Monthly Timeline")
        timeline=helper.monthly_timeline(selected_user,df)
        fig,ax=plt.subplots()
        ax.plot(timeline['time'],timeline['message'])
        plt.xticks(rotation=90)
        st.pyplot(fig)

        #Daily Timeline
        st.title("Daily Timeline")
        daily_timeline = helper.daily_timeline(selected_user, df)
        fig, ax = plt.subplots()
        ax.plot(daily_timeline['only_date'], daily_timeline['message'],color='black')
        plt.xticks(rotation=90)
        st.pyplot(fig)

        #Activity Map
        st.title("Activity Map")
        col1,col2=st.columns(2)

        with col1:
            st.header("Most Busy Day")
            busy_day=helper.week_activity_map(selected_user,df)
            fig,ax=plt.subplots()
            ax.bar(busy_day.index,busy_day.values,color='purple')
            plt.xticks(rotation=90)
            st.pyplot(fig)

        with col2:
            st.header("Most Busy Month")
            busy_month = helper.month_activity_map(selected_user, df)
            fig, ax = plt.subplots()
            ax.bar(busy_month.index, busy_month.values, color='orange')
            plt.xticks(rotation=90)
            st.pyplot(fig)

        # Heatmap
        st.title("Weekly Activity Map")
        user_heatmap=helper.activity_heatmap(selected_user,df)
        fig,ax=plt.subplots()
        ax=sns.heatmap(user_heatmap,cmap='YlOrRd')
        st.pyplot(fig)

        # Sentiment Analysis
        st.title("Sentiment Analysis")

        sent_df, sent_counts = helper.sentiment_analysis(selected_user, df)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Messages with Sentiment")
            st.dataframe(sent_df.reset_index(drop=True))

        with col2:
            st.subheader("Sentiment Distribution")
            if not sent_counts.empty:
                fig, ax = plt.subplots()
                labels = sent_counts.index
                values = sent_counts.values
                ax.pie(values, labels=labels, autopct='%.1f%%', startangle=90)
                ax.axis('equal')
                st.pyplot(fig)
            else:
                st.write("No messages to analyze")

        # 🕒 Sentiment over time
        st.markdown("---")
        st.title("📈 Sentiment Over Time")

        sent_timeline = helper.sentiment_timeline(selected_user, df)
        if not sent_timeline.empty:
            fig, ax = plt.subplots()
            ax.plot(sent_timeline['only_date'], sent_timeline['compound'], marker='o')
            plt.xticks(rotation=90)
            ax.set_ylabel("Average Sentiment")
            st.pyplot(fig)
        else:
            st.write("Not enough data for sentiment timeline.")

        # run sentiment first
        sent_df, _ = helper.sentiment_analysis(selected_user, df)
        df['sentiment'] = sent_df['sentiment']

        # 🔥 Fight Detector
        st.markdown("---")
        st.title("🔥 Fight / Argument Detector")

        fight_df = helper.fight_detector(df)

        if not fight_df.empty:
            st.write(f"⚡ Total fights detected: {fight_df.shape[0]}")
            st.dataframe(fight_df)

            # Abuse / Profanity Analysis
            st.markdown("---")
            st.title("⚠️ Abuse / Profanity Analysis")

            abuse_df = helper.abuse_analysis(selected_user, df)

            if abuse_df.shape[0] > 0:
                st.write(f"Total abusive messages: {abuse_df.shape[0]}")
                st.dataframe(abuse_df)
            else:
                st.write("No abusive messages detected.")

            # Optional short visualization
            fig, ax = plt.subplots()
            ax.bar(fight_df["Time Window Start"].dt.strftime("%d-%b %H:%M"), fight_df["Negative Msg Count"])
            plt.xticks(rotation="vertical")
            ax.set_ylabel("Negative Msg Count")
            st.pyplot(fig)

        else:
            st.write("No significant argument detected.")

        # 👇 User-wise Mood (Overall only)
        if selected_user == 'Overall':
            st.markdown("---")
            st.title("😊 User-wise Average Sentiment")

            user_mood = helper.user_sentiment_summary(df)  # table
            avg_sent = user_mood.apply(
                lambda row: (row.get('Positive', 0) - row.get('Negative', 0)) / row.sum(), axis=1
            )  # compound style score

            n_users = len(avg_sent)
            fig_height = max(6, n_users * 0.35)  # dynamic height: 0.35 inch per user (tweakable)

            # convert to numeric positions to avoid matplotlib categorical spacing issues
            y_pos = list(range(n_users))
            fig, ax = plt.subplots(figsize=(10, fig_height))

            ax.barh(y_pos, avg_sent.values, color='teal')

            # set yticks to user names
            ax.set_yticks(y_pos)
            ax.set_yticklabels(avg_sent.index, fontsize=9)  # slightly smaller font

            ax.set_xlabel("Average Sentiment (compound)")

            # prepare sentiment type labels and place them outside bars (left for negative, right for positive)
            for i, value in enumerate(avg_sent.values):
                if value > 0.05:
                    label = "Positive"
                    x_offset = 0.02
                    ha = 'left'
                elif value < -0.05:
                    label = "Negative"
                    x_offset = -0.02
                    ha = 'right'
                else:
                    label = "Neutral"
                    # place neutral a little right of the bar
                    x_offset = 0.02
                    ha = 'left'

                # Use different offset direction based on sign so label doesn't overlap the bar
                ax.text(value + x_offset, i, label, fontsize=9, va='center', ha=ha)

            plt.tight_layout()
            st.pyplot(fig)

            # summary table niche show karo
            st.subheader("📌 Sentiment Count per User")
            st.dataframe(user_mood)

        # #kon kitni baat kiya(group ke liye)
        # if selected_user=='Overall':
        #     st.title("Most Busy Users")
        #     x,new_df=helper.most_busy_users(df)
        #     fig,ax=plt.subplots()
        #
        #     col1, col2= st.columns(2)
        #
        #     with col1:
        #         ax.bar(x.index, x.values,color='red')
        #         plt.xticks(rotation=90)
        #         st.pyplot(fig)
        #     with col2:
        #         st.dataframe(new_df)

        #kon kitni baat kiya(group ke liye)
        if selected_user=='Overall':
            st.title("Most Busy Users")
            x,new_df=helper.most_busy_users(df)
            fig,ax=plt.subplots()

            col1, col2= st.columns(2)

            with col1:
                ax.bar(x.index, x.values,color='red')
                plt.xticks(rotation=90)
                st.pyplot(fig)
            with col2:
                st.dataframe(new_df)

        # 🏷️ Most Tagged Person
        st.title("Most Tagged Person")

        tagged_df = helper.most_tagged_person(selected_user, df)

        if not tagged_df.empty:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Tagged users list")
                st.dataframe(tagged_df)

            with col2:
                st.subheader("Top tagged users")
                fig, ax = plt.subplots()
                ax.barh(tagged_df['tag'].head(10), tagged_df['count'].head(10))
                plt.xticks(rotation=0)
                st.pyplot(fig)
        else:
            st.write("No tagged users (@name) found in this selection.")


    #WordCloud
    st.title("WordCloud")
    df_wc= helper.create_wordcloud(selected_user,df)
    fig,ax=plt.subplots()
    ax.imshow(df_wc)
    ax.axis('off')
    st.pyplot(fig)

    #most common words
    most_common_df=helper.most_common_words(selected_user,df)

    fig,ax=plt.subplots()
    ax.barh(most_common_df[0], most_common_df[1],color='silver')
    plt.xticks(rotation=90)
    st.title("Most Common Words")
    st.pyplot(fig)

    #Emoji analysis
    emoji_df=helper.emoji_helper(selected_user,df)
    st.title('Emoji Analysis')

    col1,col2=st.columns(2)

    with col1:
        st.dataframe(emoji_df)
    with col2:
        fig,ax=plt.subplots()
        ax.pie(emoji_df[1].head(),labels=emoji_df[0].head(),autopct='%0.2f')
        st.pyplot(fig)

    # 🔍 Most discussed topics (Topic Modeling)
    st.title("Most Discussed Topics (Hinglish Topic Modeling)")

    topics_df = helper.topic_modeling(selected_user, df)

    if not topics_df.empty:
        st.dataframe(topics_df)
    else:
        st.write("Not enough text data for topic modeling yet.")


    st.markdown("---")
    st.title("🧹 Auto Chat Cleanup Suggestions")

    if st.checkbox("Enable Auto Cleanup Suggestions"):
        with st.spinner("Generating suggestions..."):
            suggestions_df = helper.generate_cleanup_suggestions(df)

        if suggestions_df.empty:
            st.write("No cleanup suggestions found for this chat.")
        else:
            st.write(f"Total suggestions: {suggestions_df.shape[0]}")
            # show table
            st.dataframe(suggestions_df.reset_index(drop=True))

            # allow filtering by issue type
            issue_types = ['All'] + sorted(suggestions_df['issue_type'].dropna().unique().tolist())
            sel_type = st.selectbox("Filter by issue type", issue_types)
            if sel_type != 'All':
                view_df = suggestions_df[suggestions_df['issue_type'] == sel_type]
            else:
                view_df = suggestions_df

            st.dataframe(view_df.reset_index(drop=True))

            # Download suggestions CSV
            csv = view_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download suggestions (CSV)", csv, file_name="cleanup_suggestions.csv", mime="text/csv")

            # Optional: let user apply selected suggestions to generate a 'cleaned' chat export
            if st.button("Generate cleaned chat (remove forwarded & exact duplicates)"):
                # Simple cleaning: remove forwarded and exact duplicates (norm match)
                to_remove_idx = set()

                fw_df = helper.detect_forwarded_messages(df)
                # remove exact duplicate groups with duplicate_group != 0
                dup_df = helper.detect_duplicate_messages(df, similarity_threshold=0.95)  # stricter

                # mark rows to remove by matching message text & date & user
                # We'll do a simple approach: remove messages present in fw_df OR if duplicate_group present (keep 1 per group)
                if not fw_df.empty:
                    # build set of message signatures
                    fw_sigs = set((r['user'], str(r['date']), str(r['message'])) for _, r in fw_df.iterrows())
                else:
                    fw_sigs = set()

                cleaned = df.copy().reset_index(drop=True)
                # exact duplicate removal: keep first occurrence for each normalized text
                if not dup_df.empty:
                    # get duplicate normalized messages groups
                    # here detect_duplicate_messages in helper returns message groups (with group ids)
                    dup_rows = helper.detect_duplicate_messages(df, similarity_threshold=0.95)
                    # we will drop all messages that are in dup_rows except one per group
                    keep_signatures = set()
                    # for each group, keep the earliest date
                    for gid, group in dup_rows.groupby('duplicate_group'):
                        # pick earliest to keep
                        first = group.sort_values('date').iloc[0]
                        keep_signatures.add((first['user'], str(first['date']), first['message']))

                    # now form remove set: all dup rows that are not in keep_signatures
                    for _, r in dup_rows.iterrows():
                        sig = (r['user'], str(r['date']), r['message'])
                        if sig not in keep_signatures:
                            fw_sigs.add(sig)  # reuse fw_sigs as remove set


                # filter cleaned
                def sig_of_row(r):
                    return (r['user'], str(r['date']), r['message'])


                mask_keep = [sig_of_row(row) not in fw_sigs for _, row in cleaned.iterrows()]
                cleaned = cleaned[mask_keep]

                # produce downloadable cleaned chat text (using preprocessor reverse if you have, else simple join)
                # simple export: produce csv of cleaned messages
                cleaned_csv = cleaned.to_csv(index=False).encode('utf-8')
                st.download_button("Download cleaned chat (CSV)", cleaned_csv, file_name="cleaned_chat.csv",
                                   mime="text/csv")

    # 🔥 Anger Diffusion Map
    st.markdown("---")
    st.title("🌋 Anger Diffusion Map")

    if 'sentiment' not in df.columns:
        st.write("Please run Sentiment Analysis first.")
    else:
        diffusion_df = helper.anger_diffusion_map(df)

        if diffusion_df.empty:
            st.write("No anger diffusion detected.")
        else:
            st.write("Detected anger spread patterns:")
            st.dataframe(diffusion_df)

            # Plot top 5 longest negative diffusion events
            top = diffusion_df.sort_values("Duration (mins)", ascending=False).head(5)

            fig, ax = plt.subplots(figsize=(10, 4))
            ax.barh(top["Users in Order"], top["Duration (mins)"], color='red')
            ax.set_xlabel("Duration (minutes)")
            ax.set_ylabel("User Spread Order")
            plt.tight_layout()
            st.pyplot(fig)

    # 🏆 Chat Awards
    st.markdown("---")
    st.title("🏆 Chat Awards")

    awards = helper.chat_awards(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("💛 Most Supportive")
        st.write(awards.get("Most Supportive", "—"))

    with col2:
        st.subheader("🤣 Funniest")
        st.write(awards.get("Funniest", "—"))

    with col3:
        st.subheader("🤫 Silent Reader")
        st.write(awards.get("Silent Reader", "—"))



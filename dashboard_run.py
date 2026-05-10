# necessary imports 
import re
import streamlit as st
import pandas as pd
import networkx as nx
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Trend of Comment Behavior of Subreddit r/meirl",
    layout="wide"
)

@st.cache_data

def load_data():
    # read in initial dataset (very large, cannot push files directly to GitHub so have to insert/remove before using/pushing respectively)
    posts = pd.read_csv("meirl_datasets/the-reddit-irl-dataset-posts.csv")
    comments = pd.read_csv("meirl_datasets/the-reddit-irl-dataset-comments.csv")

    # data cleaning
    posts = posts.dropna(subset=['url'])
    comments = comments.dropna(subset=['sentiment'])

    # create dictionary containing all post IDs, with the key being the number of comments
    # Function to extract post_id from permalink
    def extract_post_id(permalink):
        if isinstance(permalink, str):
            match = re.search(r'/comments/([^/]+)/', permalink)
            if match:
                return match.group(1)
        return None

    # Extract post_id and create a new column in comments DataFrame
    comments['post_id'] = comments['permalink'].apply(extract_post_id)

    # Filter out comments where post_id could not be extracted (e.g., malformed permalinks)
    comments_with_valid_post_id = comments.dropna(subset=['post_id'])

    # Group by post_id and count comments efficiently
    comment_counts_per_post = comments_with_valid_post_id.groupby('post_id')['id'].count()

    # Initialize post_dict with all unique post IDs from the posts DataFrame, setting count to 0 initially
    post_dict = {pid: 0 for pid in posts['id'].unique()}

    # Update counts for posts that have comments
    # Only update if post_id exists in our original posts list from 'posts' DataFrame
    for post_id, count in comment_counts_per_post.items():
        if post_id in post_dict:
            post_dict[post_id] = count

    # convert earlier dict into a dataframe
    df = pd.DataFrame(list(post_dict.items()), columns=['post_id', 'num_comments'])

    # exclude posts with no comments
    df = df[df['num_comments'] > 0]

    # create appropriate intertvals
    bins = [0, 1, 5, 10, 50, 100, 500, df['num_comments'].max()]
    labels = ['1', '2-5', '6-10', '11-50', '51-100', '101-500', '500+']

    # put into bins
    df['comment_range'] = pd.cut(df['num_comments'], bins=bins, labels=labels)

    # calculate percentage of posts in each bin
    distribution = df['comment_range'].value_counts(normalize=True) * 100

    # sort for nicer output
    distribution = distribution.sort_index()

    # make a sample dataframe of 10,000 posts to use for analysis by using the ratio from above
    # total sample size
    sample_size = 10000
    # sample post dictionary
    sp_dicts = []

    # loop to go through each interval
    for bin_label, proportion in distribution.items():

        #number to sample from this bin
        n = int((proportion / 100) * sample_size)

        #get all the posts within this interval in this bin
        bin_df = df[df['comment_range'] == bin_label]

        #sample from the bins into a new dict
        sampled_bin = bin_df.sample(n=min(n, len(bin_df)), random_state=42)
        sp_dicts.append(sampled_bin)

    # combine all sampled bins
    sp_dict = pd.concat(sp_dicts)

    # shuffle to see variation in number of comments in output
    sp_dict = sp_dict.sample(frac=1, random_state=42).reset_index(drop=True)

    # separate the comments that apply to these posts to use for analysis
    # sampled post id
    sp_ids = sp_dict['post_id']

    # filter
    sampled_comments = comments[comments['post_id'].isin(sp_ids)]

    # setting up the posts dataframe with the other attributes
    # get the sample posts from posts
    sampled_posts_only = posts[posts['id'].isin(sp_ids)]

    # can change based on what we need
    post_columns = [
        'id',
        'created_utc',
        'permalink',
        'score',
    ]

    # copy to a new dataframe with the columns we need for analysis
    sampled_posts_clean = sampled_posts_only[post_columns].copy()

    # setting up the posts dataframe with the other attributes
    # attributes we take from original dataframe
    comment_columns = [
        'id',
        'created_utc',
        'permalink',
        'body',
        'sentiment',
        'post_id'
    ]

    # copy into new comments dataframe for analysis
    sampled_comments_clean = sampled_comments[comment_columns].copy()
    
    return sampled_posts_clean, sampled_comments_clean, distribution, sp_dict

@st.cache_data
def prepare_graph_data():
    sampled_posts_clean, sampled_comments_clean, distribution, sp_dict = load_data()

    network1 = nx.Graph()

    # add post nodes
    for _, row in sampled_posts_clean.iterrows():
        network1.add_node(
            row["id"],
            type="post",
            created_utc=row["created_utc"],
            permalink=row["permalink"],
            score=row["score"]
        )

    valid_post_ids = set(sampled_posts_clean["id"])

    # add comment nodes and connect to posts
    for _, row in sampled_comments_clean.iterrows():
        comment_id = row["id"]
        post_id = row["post_id"]
        sentiment = row["sentiment"]

        if post_id in valid_post_ids:
            if sentiment > 0.2:
                color = "green"
            elif sentiment < -0.2:
                color = "red"
            else:
                color = "blue"

            network1.add_node(
                comment_id,
                type="comment",
                created_utc=row["created_utc"],
                permalink=row["permalink"],
                body=row["body"],
                sentiment=sentiment,
                post_id=post_id,
                color=color
            )
            network1.add_edge(post_id, comment_id)

    return sampled_posts_clean, sampled_comments_clean, distribution, sp_dict, network1   

st.title("Trend of Comment Behavior of Subreddit r/meirl")
st.write(
    """Exploration of comment behavior of posts from subreddits r/meirl and r/me_irl,
        which are both considered "relatable humor" subreddits, looking at posts and the types 
        of comments attached to it. Here, we will see what kinds of comments appear on posts, 
        and what kinds of comments appear over time.
        In the tabs below, you can investigate the post and comment attributes, network structure,
        design and metrics, comment behavior over time for top posts in the subreddits, and conclusions
        from our investigations!
    """
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Graph Structure",
    "Sentiment Over Time",
    "Interpretation & Conclusions"
])

# page 1
with tab1:
    st.header("Overview")
    st.markdown("""
    The main idea of this page is to give you an introduction to the dataset! Our datasets come from researchers on Kaggle,
    in which they scrolled relateable humor subreddits on Reddit.com, namely r/meirl and r/me_irl, and collected posts and their
    comments in order to ask interesting questions about their behavior. While relatively niche, these subreddits in particular are
    interesting to look at because it exists within the "humor" scope, and it can pose some interesting questions
    when a lot of negativity is seen, and what that can tell us about commenting behaior on Reddit for these communities.
    You will see how many post nodes and comment nodes (that correspond to a post node) will exist in our network, where we dive deeper
    into what a post's comment section can look like in regard to "sentiment."
    
    *Sentiment* is what the researchers who collected this datset as the overall tone of a comment, with
    the value of a comment's sentiment existing between 1 (completely positive comment) and -1 (completely negative comment.)
    While sentiment isn't used usefully in this tab, it becomes more important to look at in the later tabs, as we are able to determine what a 
    post's comment section overall tone is and how it can impact what kinds of comments occur over time.          
    """)
    # load in data here
    post_nodes, comment_nodes, dist, sp_dict = load_data()
    average_comments_per_post = sp_dict['num_comments'].mean()
    
    st.markdown("""
    ## Research Questions
       Below are the questions that are guiding our research, and what helps us structure our network configurations, graphs, and analysis:
       #### 1. Do comments from the same post share the same sentiment?
       #### 2. Regarding sentiment, how does a post's comment section change over time?
       #### 3. Do posts that already have a lot of comments tend to attract more over time?        
    """)

    st.markdown("""
    ## Dataset Details
    """)
    c1, c2, c3 = st.columns(3)
    c1.metric("Post Nodes", len(post_nodes))
    c2.metric("Comment Nodes", len(comment_nodes))
    c3.metric("Average No. of Comments Per Post", round(average_comments_per_post, 3))
    st.markdown("""
    ### Comment Distribution Among Posts
    The comment distribution shows the proportion of posts (shown on the left have side) that have
    a given number of comments (shown on the right side.) This can help give an idea of what the post nodes
    in the network will look like, with more posts having fewer comments and fewer posts having a lot of comments.
            
    This also gives a sneak-peak into what the degree distribution will look like, detailed more in the next tab...
    """)
    dist = dist.to_frame()
    # distribution dropdown
    with st.expander("Comment Distribution Dropdown"):
        st.dataframe(
            dist[[
                "proportion" 
            ]],
        use_container_width=True
        )

    
    st.markdown("""
    ### Post Nodes
    The post nodes in our network are representative of the posts found in the r/meirl and r/me_irl
    subreddits. While cut down after cleaning, we have just under 10,000 posts that all have comments attached to them, as
    posts that have no comments within this dataset are not very useful to the questions we are trying to answer.
                
    Each post node has four attributes of interest:
    1. 'id': the unique indentification value of the post
                
    2. 'created_utc': the timestamp that indicates when the post was posted in the subreddit it belongs to
    
    3. 'permalink': the link to the post on Reddit.com
                
    4. 'score': the cumulative number of upvotes/downvotes on a post
                
    Following below is a dropdown of all the posts intended for our network, along with their attributes:
    """)
    # post node dropdown
    with st.expander("Post Nodes Dropdown"):
        st.dataframe(
            post_nodes[[
                "id", "created_utc", "permalink", "score"
            ]],
        use_container_width=True
        )

    st.markdown("""
    ### Comment Nodes
    The comment nodes in our network represent the comments on a respective post found in the subreddits. This is a significantly
    larger dataset, with over 69,000 comments, each being connected to exactly one post. The attributes of interest for the comment nodes
    are:
    1. 'id': the unique indentification value of the comment
    
    2. 'created_utc': the timestamp that indicates when the comment was posted on the post it belongs to
                
    3. 'permalink': the link to the comment on Reddit.com
                
    4. 'body': the actual contents of the comment itself / what the comment says
                
    5. 'sentiment': the sentiment score (described above) of comment, indicating where on a scale (-1 to 1) the tone of a comment lies
                
    6. 'post_id': the identification value of the post the comment is posted on

    Following below is a dropdown of all the comments intended for our network, along with their attributes:          
    """)
    # comment node dropdown
    with st.expander("Comment Nodes Dropdown"):
        st.dataframe(
            comment_nodes[[
                "id", "created_utc", "permalink", "body", "sentiment", "post_id"
            ]],
        use_container_width=True
        )

    


# page 2 - Graph Structure
with tab2:
    st.header("Graph Structure & Degree Distribution")

    sampled_posts_clean, sampled_comments_clean, distribution, sp_dict, network1 = prepare_graph_data()

    st.markdown("""
    This tab shows two network-related views:

    - a bipartite subgraph
    - a degree distribution graph
    """)

    num_posts_to_show = st.slider(
        "Number of post nodes to show in the bipartite subgraph",
        min_value=1,
        max_value=min(25, len(sampled_posts_clean)),
        value=5
    )

    st.subheader("Bipartite Subgraph: Posts and Comments")
    st.markdown("""
    Black nodes represent posts. Colored nodes represent comments attached to one post.
    Comment color reflects sentiment grouping:
    - green = more positive
    - red = more negative
    - blue = more neutral
    """)

    # choose some post nodes
    post_nodes_subset = [
        n for n, d in network1.nodes(data=True) if d["type"] == "post"
    ][:num_posts_to_show]

    # collect selected posts and their connected comments
    nodes_to_draw = set(post_nodes_subset)
    for p in post_nodes_subset:
        nodes_to_draw.update(network1.neighbors(p))

    # create subgraph
    small_graph = network1.subgraph(nodes_to_draw)

    # separate node sets
    post_nodes_small = [n for n, d in small_graph.nodes(data=True) if d["type"] == "post"]
    comment_nodes_small = [n for n, d in small_graph.nodes(data=True) if d["type"] == "comment"]

    # bipartite layout
    bipartite_pos = nx.bipartite_layout(small_graph, post_nodes_small)

    # colors and sizes
    node_colors = []
    node_sizes = []

    for n in small_graph.nodes():
        if small_graph.nodes[n]["type"] == "post":
            node_colors.append("black")
            node_sizes.append(250)
        else:
            node_colors.append(small_graph.nodes[n].get("color", "blue"))
            node_sizes.append(80)

    fig1, ax1 = plt.subplots(figsize=(12, 8))
    nx.draw(
        small_graph,
        bipartite_pos,
        node_color=node_colors,
        node_size=node_sizes,
        with_labels=False,
        ax=ax1
    )
    ax1.set_title("Bipartite Subgraph: Posts and Comments")

    st.pyplot(fig1)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Nodes", small_graph.number_of_nodes())
    c2.metric("Subgraph Edges", small_graph.number_of_edges())
    c3.metric("Post Nodes", len(post_nodes_small))
    c4.metric("Comment Nodes", len(comment_nodes_small))

    st.subheader("Degree Distribution of Post Nodes")

    degrees = sp_dict["num_comments"]
    max_degree = int(degrees.max())
    bins = range(0, max_degree + 10, 10)

    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.hist(degrees, bins=bins)
    ax2.set_xlabel("Number of Comments")
    ax2.set_ylabel("Number of Posts")
    ax2.set_title("Degree Distribution of Post Nodes")

    st.pyplot(fig2, use_container_width=True)

    st.markdown("""
    This graph above shows the degree distribution of the post nodes, which is equivalent to the distribution of the number of comments per post.
    We can see that most posts have a small number of comments, while a few posts have a large number of comments, which is consistent with the comment distribution we saw in the
     first tab. This is also indicative of the standard power law distrbution that is commonly seen in real-world networks, with 
    a heavy side on the left tailing off to the right ("rich get richer," small amounts of the posts contain most
    of the comments in the network.)
    """)


# page 3 - Sentiment Over Time
with tab3:
    st.header("Sentiment Over Time")

    #get data from above
    posts, comments, _, sp_dict = load_data()
    #convert the utc into readable timestamps
    comments["created_dt"] = pd.to_datetime(comments["created_utc"], unit="s")

    #sort comments with the posts they belong to, and calculate the sentiment of the post by adding 
    # the comment sentiments and merge with the existing number of comments attribute 
    post_summary = comments.groupby("post_id")["sentiment"].sum().reset_index()
    post_summary = post_summary.merge(sp_dict[["post_id", "num_comments"]], on="post_id")

    #sort to find the posts with the most comments  
    top_posts = post_summary.sort_values("num_comments", ascending=False).head(20)
    #TEXT
    st.markdown("""
    ### Network Graph

    Select a post from the dropdown menu to study. The posts are arranged based on the 
    number of comments they have, with the top being the post with the most comments. 
    Slide to change the munber of comments shown in the plots. 

    The color of the nodes signifies the sentiment score of the nodes as defined in the dataset, 
    with blue being neutral, green being positive, and red being negative.
    Notice how the sentiment of the middle(post) node changes as number of the comments increase.
    """)

    #user can select which post to view from a dropdown menu, sorted in decreasing order of the number of comments 
    selected_post = st.selectbox("Post", top_posts["post_id"])

    #for the selected post, extract it's comments and sort by the timestamp 
    post_comments = comments[comments["post_id"] == selected_post].sort_values("created_dt").reset_index(drop=True)

    if len(post_comments) == 0:
        st.warning("No comments")
    else:
        #set the maximum comment to be chosen to either the maximum number
        # comments or 800 if it's more than that to keep it visible 
        max_n = min(800, len(post_comments))

        #user can pick how many comments to show in plot using slider min 1, default 5
        step = st.slider("Comments shown", 1, max_n, 20)

        #takes the N amount of comments from the sorted dataframe
        current = post_comments.iloc[:step] 

        #take the sentiment of the comments and add to find the curr post sentiment 
        sentiment = current["sentiment"].sum() 

        #graph
        G = nx.Graph()
        G.add_node(selected_post)

        #color coding node based on sentiment score 
        for _, r in current.iterrows():
            color = "green" if r["sentiment"] > 0 else "red" if r["sentiment"] < 0 else "blue"
            G.add_node(r["id"], color=color)
            G.add_edge(selected_post, r["id"])

        pos = nx.spring_layout(G, seed=42)

        edge_x, edge_y = [], []
        for e in G.edges():
            x0, y0 = pos[e[0]]
            x1, y1 = pos[e[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x, node_y, node_c = [], [], []
        for n in G.nodes():
            x, y = pos[n]
            node_x.append(x)
            node_y.append(y)
            if n == selected_post: # color coding of the post node
                node_c.append("green" if sentiment > 0 else "red" if sentiment < 0 else "blue")
            else:
                node_c.append(G.nodes[n].get("color", "blue"))

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(color="gray"), showlegend=False))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers",
                                 marker=dict(size=12, color=node_c),
                                 showlegend=False))

        fig.update_layout(height=400, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis=dict(visible=False), yaxis=dict(visible=False))

        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        ### Line Chart

        This chart shows how the overall post sentiment changes over the number of comments.
        The comments are picked based on their timestamps, with the first comments shown 
        being the ones that are commented before the rest.   
        """)


        # plot
        df = current.copy()
        df["n"] = range(1, len(df) + 1)
        df["cum_sent"] = df["sentiment"].cumsum()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["n"],
            y=df["cum_sent"],
            mode="lines",
            name="Post sentiment"
        ))

        fig2.update_layout(
            height=400,
            xaxis=dict(title="Number of comments"),
            yaxis=dict(title="Overall post sentiment"),
            title="Post Sentiment Over Number of Comments"
        )

        st.plotly_chart(fig2, use_container_width=True)



# page 4 - Conclusion
with tab4:
    st.header("Conclusions")

    # Research Question 1
    st.markdown("""
        ### Comment Sentiment Variance
        """)
    
    post_std = []
    post_avg_sentiments = []

    for post_id in sampled_posts_clean['id']:
        post_comments = sampled_comments_clean[sampled_comments_clean['post_id'] == post_id]

        if len(post_comments) < 2:
            # Need at least 2 comments for standard deviation, but can calculate average sentiment
            if len(post_comments) == 1:
                post_avg_sentiments.append(post_comments['sentiment'].iloc[0])
            continue

        standard_dev = post_comments['sentiment'].std()
        post_std.append(standard_dev)

        # Calculate average sentiment for comments in the current post
        avg_post_sentiment = post_comments['sentiment'].mean()
        post_avg_sentiments.append(avg_post_sentiment)

    # average standard deviation across posts
    avg_std = sum(post_std) / len(post_std)

    # average sentiment score across all posts
    if post_avg_sentiments:
        overall_avg_sentiment = sum(post_avg_sentiments) / len(post_avg_sentiments)
    else:
        overall_avg_sentiment = 0.0 # Handle case where no posts have comments


    d1, d2 = st.columns(2)
    d1.metric("Average Comment Sentiment Per Post", round(overall_avg_sentiment, 5))
    d2.metric("Average Sentiment Standard Deviation Per Post", round(avg_std, 5))
    
    st.markdown("""
        For our first research question, we unfortunately cannot draw any conclusions regarding if
        comments within a post's comment section tend to share the same sentiment. Because the average 
        sentiment per post is 0.15, meaning most comments trend slightly postive, and the
        average standard deviation is almost double that, it implies that most of the comment's sentiments
        per post are very spread out from the average sentiment. Therefore, we cannot conclude that there is a 
        distinctive pattern to demostrate that all comments on a post tend to be around the same sentiment.
        """)
    
    # Research Question 2
    st.markdown("""
        ### Sentiment Over Time - Trends
        
        For our second research question, we can make some more interesting conclusions
        about the sentiment over a post's comment section lifespan. By looking at the five top-commented
        positive sentiment posts and the five top-commented negative sentiment posts, we can see how the 
        sentiment of the comments starts out and how it ends. Overall, we see:
            
        * 3 out of 5 postive sentiment posts start with mostly negative comments
        * 1 out of 5 positive sentiment posts start with mostly positive comments
        * 1 out of 5 positive sentiment posts start with neutral comments
        * 5 out of 5 negative sentiment posts start with negative comments
                
        While it is a small portion, seeing that the top posts for both positive and negative
        sentiment comment sections start off with mostly negative comments is an interesting trend,
        but makes it diffcult to conclude anythong for positive sentiment posts. However, since all
        the top negative comment section posts started negative and remained negative, this means that
        there could be interest in researching if early negative behvaior in comment sections
        sets the tone for how the comment section "ends."
        """)
    
    # Comment Growth Over Time with Sentiment Changing Line Color

    post_sentiment_summary = sampled_comments_clean.groupby('post_id')['sentiment'].sum().reset_index()
    post_sentiment_summary.columns = ['post_id', 'total_sentiment']

    # merge with number of comments from sp_dict
    post_sentiment_summary = post_sentiment_summary.merge(
        sp_dict[['post_id', 'num_comments']],
        on='post_id',
        how='left'
    )
    
    top_positive_posts = post_sentiment_summary[post_sentiment_summary['total_sentiment'] > 0] \
        .sort_values('num_comments', ascending=False) \
        .head(5)

    #high-comment negative posts
    top_negative_posts = post_sentiment_summary[post_sentiment_summary['total_sentiment'] < 0] \
        .sort_values('num_comments', ascending=False) \
        .head(5)


    comparison_post_ids = top_positive_posts['post_id'].tolist() + top_negative_posts['post_id'].tolist()
    print(comparison_post_ids)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot number of comments over time for several posts
    for post_id in comparison_post_ids:
        post_comments = sampled_comments_clean[
            sampled_comments_clean["post_id"] == post_id
        ].copy()

        post_comments = post_comments.sort_values("created_utc")

        post_comments["comment_number"] = range(1, len(post_comments) + 1)
        post_comments["cumulative_sentiment"] = post_comments["sentiment"].cumsum()

        x = post_comments["created_utc"].tolist()
        y = post_comments["comment_number"].tolist()
        s = post_comments["cumulative_sentiment"].tolist()

        for i in range(len(x) - 1):
            if s[i] > 0:
                segment_color = "green"
            elif s[i] < 0:
                segment_color = "red"
            else:
                segment_color = "blue"

            ax.plot(
                [x[i], x[i + 1]],
                [y[i], y[i + 1]],
                color=segment_color,
                linewidth=2,
                alpha=0.8
            )

    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Number of Comments")
    ax.set_title("Comment Growth Over Time with Sentiment-Based Line Color")

    fig.text(
        0.5,
        -0.05,
        "Line color shows cumulative sentiment at that point: green = positive, red = negative, blue = neutral.",
        ha="center",
        fontsize=10
    )

    plt.tight_layout()

    # Streamlit display
    st.pyplot(fig)

    # Research Question 3
    st.markdown("""
        ### Early Comment Growth vs. Final Popularity
        
        For our last research question, we found the number of comments obtained within the 
        first hour of the post's creation, and plotted it against the total of number of comments
        that post received overall. Our plan was to see if there was any clear trend, particularly a linear upwards trend,
        which would imply that posts that started with a lot of comments in the first hour 
        would continue and end with a large amount of comments.
                
        However, this was not the case, as it seemed a majority of posts started with 0-10 comments
        within the first hour, regardless of the total number of comments it received in the end. And, interestingly
        enough, the few posts that did start off with a lot of comments actually had less overall
        comments. Therefore, while we weren't able to conclude any linear pattern, it is interesting to see that
        early commenting behavior does not have much of an impact for how popular the post becomes later on.
        """)
    
    # scatterplot of Early Growth vs. Final Popularity here

    early_counts = []
    final_counts = []

    for post_id in sampled_posts_clean['id']:
        post_comments = sampled_comments_clean[sampled_comments_clean['post_id'] == post_id].copy()

        # skip posts that are too little
        if len(post_comments) < 5:
            continue

        post_comments = post_comments.sort_values('created_utc')

        # time since first comment
        post_comments['time_since_start'] = (
            post_comments['created_utc'] - post_comments['created_utc'].iloc[0]
        )

        # count how many comments occur in the first hour
        early_count = len(post_comments[post_comments['time_since_start'] <= 3600])

        # total comments
        total_comments = len(post_comments)

        early_counts.append(early_count)
        final_counts.append(total_comments)

    # graph

    fig, ax = plt.subplots(figsize=(8, 6))  # slightly smaller

    ax.scatter(early_counts, final_counts, alpha=0.5)

    ax.set_xlabel("Number of comments in first hour")
    ax.set_ylabel("Total number of comments")
    ax.set_title("Early Growth vs Final Popularity")

    # Add whitespace inside figure
    plt.subplots_adjust(left=0.4, right=1, top=0.8, bottom=0.2)

    col1, col2, col3 = st.columns([1, 6, 1])  # middle column is wider

    with col2:
        st.pyplot(fig, use_container_width=True)
    

    st.header("Limitations")
    st.markdown("""
        One of the largest limitations of this dataset is the lack of user IDs; while it was done for
        anonymity purposes, it is frustrating that it is omitted for this data set since it severly limited
        a lot of the other social network analyses we wuld be able to look at, such as:
        
        * Seeing who the top posters (by users whose posts had lots of comments) are
        * If users who tend to post comments of a certain sentiment interact with other users who post of the same sentiment
        * Analyzing time stamps of user's commenting behavior over time over multiple posts

        It also prevented us from performing any centrality measures or community detection, which was disappointing as we 
        spent a large portion of this course woking with such data.

        Another drawback, which is also an ethical concern, is the bias that is likely present when researchers determined
        the "sentiment" of a comment. Since humor is subjective, especially on a platform such as Reddit, 
        what may be seen by average researchers or viewers as negative may just be an attempt at satire or 
        sarcasm, and neutral comments may be images that aren't properly analyzed for sentimental (positive or negative) 
        context. While speculations could be made, the researchers gave no clear guidelines they used when determining
        the sentiment score of a comment (whether it be algorithmic or otherwise), so it must be concldued that some kind
        of bias must be present. However, this could also be opprtunity for any future research done to highlight more 
        detailed procedures of comment analysis for sentiment, as well as taking into account user IDs (even if codenames /
        fake identification are to be used for anonymity purposes.)
    """)
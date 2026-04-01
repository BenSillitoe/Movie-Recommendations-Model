#Imports
import pandas as pd
import numpy as np
from  sklearn.metrics.pairwise import cosine_similarity
import tkinter as tk
from tkinter import ttk
import webbrowser

#Loading Dataset
ratings = pd.read_csv('ratings.csv')
ratings = ratings.rename(columns={'userId': 'userID',
                         'movieId' : 'movieID'})

movies = pd.read_csv('movies.csv')
movies = movies.rename(columns={'movieId': 'movieID'})

links = pd.read_csv('links.csv')
links = links.rename(columns={'movieId': 'movieID'})

#intergrate the links from IMDB to the interface
links['imdb_link'] = 'https://www.imdb.com/title/tt' + links['imdbId'].astype(str).str.zfill(7)
movies =movies.merge(links[['movieID', 'imdb_link']], on='movieID', how='left')

avg_ratings = ratings.groupby('movieID')['rating'].mean().round(1).reset_index()
avg_ratings.columns = ['movieID', 'avg_rating']
movies = movies.merge(avg_ratings, on='movieID', how='left')

#User-movie table
user_movie = ratings.pivot_table(
    index = 'userID',
    columns = 'movieID',
    values = 'rating'
)

#Handling missing values
user_movie_filled = user_movie.apply(lambda row: row.fillna(row.mean()), axis=1)

#Similarity Evaluation
movie_similarity = cosine_similarity(user_movie_filled.T)

#Converting to DataFrame
movie_similarity_df = pd.DataFrame(
    movie_similarity,
    index = user_movie.columns,
    columns = user_movie.columns
)

#Recommendation Function
def recommend_movies(movie_id, num_recommendations = 5, genre_filter= None, min_rating=None):
    if movie_id not in movie_similarity_df.columns:
        raise ValueError(f"Movie ID {movie_id} not found in dataset")

    similar_scores = movie_similarity_df[movie_id].sort_values(ascending=False)

    top_movies = similar_scores.iloc[1:9742].index
    results = movies[movies['movieID'].isin(top_movies)]

    if genre_filter:
        results = results[results["genres"].str.contains(genre_filter, case=False, na=False)]


    if min_rating:
        results = results[results['avg_rating'] >= float(min_rating)]

    n = min(num_recommendations, len(results))

    if n == 0:
        print('No movies found matching your filters')
        return movies[['title', 'genres', 'avg_rating','imdb_link']].head(0)

    results = results.sample(n=n).reset_index(drop=True)
    results = results[['title', 'genres', 'avg_rating', 'imdb_link']]

    return results


#Function for pop-up window
def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry(f"{width}x{height}+{x}+{y}")

#Card builder for nicer UI compared to the table view
def build_cards(parent, dataframe):
    for widget in parent.winfo_children():
        widget.destroy()

    #Handles empty results
    if dataframe.empty:
        tk.Label(parent, text="No movies found your filters",
                 font=('Helvetica', 12), bg='#1a1a2e', fg='#a8b2d8').pack(pady=20)
        return

    for _, record in dataframe.iterrows():
        #Extract each feild
        title = record['title']
        genres = record['genres']
        rating = record['avg_rating']
        imdb_url = record['imdb_link']

        #Card frame per film
        card = tk.Frame(parent, bg='#16213e', pady=10, padx=14)
        card.pack(fill='x', padx=12, pady=5)

        #title lable
        tk.Label(card, text=title, font=('Helvetica', 13, 'bold'),
                 bg='#16213e', fg='white', anchor='w',
                 wraplength=700, justify='left').pack(anchor='w')

        #Updated genre seperator
        genre_list = genres.replace('|', '  •  ')
        tk.Label(card, text=genre_list, font=('Helvetica', 10),
                 bg='#16213e', fg='#a8b2d8', anchor='w').pack(anchor='w', pady=2)

        # make a footer row at the bottom of the card
        footer = tk.Frame(card, bg='#16213e')
        footer.pack(fill='x', pady=4)

        # work out how many stars to show based on the rating
        stars = '★' * int(round(rating)) + '☆' * (5 - int(round(rating)))

        # show the stars and the number score next to each other
        tk.Label(footer, text=f"{stars}  {rating} / 5",
                 font=('Helvetica', 11), bg='#16213e', fg='#E8A838').pack(side='left')

        # button that opens imdb in the browser when clicked
        url = imdb_url
        tk.Button(footer, text='Open IMDB', font=('Helvetica', 10),
                  bg='#e94560', fg='white', relief='flat',
                  cursor='hand2', padx=10, pady=3,
                  command=lambda u=url: webbrowser.open(u)).pack(side='right')

        # thin line to separate each card visually
        tk.Frame(parent, bg='#0f3460', height=1).pack(fill='x', padx=12)

#displays the UI to the user
def show_ui():
    window = tk.Tk()
    window.title('Movie Recommendations')
    window.configure(bg='#1a1a2e')
    center_window(window,900, 700)

    #top section with the window title and subtitle
    header = tk.Frame(window, bg='#1a1a2e', pady=15)
    header.pack(fill='x', padx=20)
    tk.Label(header, text="Movie Recommendations",
             font=('Helvetica', 20, 'bold'),
             bg='#1a1a2e', fg='#e94560').pack(anchor='w')
    tk.Label(header, text="Filter by genre or rating, then refresh for new picks",
             font=('Helvetica', 11), bg='#1a1a2e', fg='#a8b2d8').pack(anchor='w')

    #bar across the top for the fiter controls
    filter_frame = tk.Frame(window, bg='#16213e', pady=10,padx=14)
    filter_frame.pack(fill='x', padx=20, pady=5)

    #genre text boxes
    tk.Label(filter_frame, text= 'Genre:', font=('Helvetica', 11),
             bg='#16213e', fg='#a8b2d8').pack(side='left', padx=5)
    genre_entry = tk.Entry(filter_frame, width=16, font=('Helvetica', 11),
                           bg='#0f3460', fg='white', insertbackground='white',
                           relief='flat', bd=5)
    genre_entry.pack(side='left', padx=5)

    #Ratings drop down selection
    tk.Label(filter_frame, text='Minimum Rating:', font=('Helvetica', 11),
             bg='#16213e', fg='#a8b2d8').pack(side='left', padx=10)
    rating_var = tk.StringVar(value='Any')
    rating_dropdown = ttk.Combobox(filter_frame, textvariable=rating_var, width=5,
                                   values=['Any', '1', '1.5', '2', '2.5', '3', '3.5', '4', '4.5', '5'],
                                   font=('Helvetica', 11))
    rating_dropdown.pack(side='left', padx=5)

    # runs when apply is clicked, passes the current filter values
    def apply_filter():
        genre = genre_entry.get().strip()
        min_rating = None if rating_var.get() == 'Any' else rating_var.get()
        refresh_cards(genre, min_rating)

    # runs when refresh is clicked, keeps the same filters but gets new random picks
    def refresh():
        genre = genre_entry.get().strip()
        min_rating = None if rating_var.get() == 'Any' else rating_var.get()
        refresh_cards(genre, min_rating)

    # styled apply and refresh buttons
    tk.Button(filter_frame, text='Apply', font=('Helvetica', 11, 'bold'),
        bg='#e94560', fg='white', relief='flat', padx=12, pady=4,
        highlightbackground='#e94560', highlightthickness=0,
        cursor='hand2', command=apply_filter).pack(side='left', padx=8)
    tk.Button(filter_frame, text='Refresh', font=('Helvetica', 11),
        bg='#0f3460', fg='white', relief='flat', padx=12, pady=4,
        highlightbackground='#e94560', highlightthickness=0,
        cursor='hand2', command=refresh).pack(side='left', padx=4)

    #small label to say how many results are showing
    count_label = tk.Label(window, text="", font=('Helvetica',10),
                           bg='#1a1a2e', fg='#a8b2d8')
    count_label.pack(anchor='w', padx=20)

    #container that holds the scrollable aread
    container = tk.Frame(window, bg= '#1a1a2e')
    container.pack(fill='both', expand=True, padx=20, pady=5)

    # canvas so that the cards can scroll up and down
    canvas = tk.Canvas(container, bg='#1a1a2e', highlightthickness=0)
    scrollbar = ttk.Scrollbar(container, orient='vertical', command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)
    scrollbar.pack(side='right', fill='y')
    canvas.pack(side='left', fill='both', expand= True)

    scrollable_frame = tk.Frame(canvas, bg='#1a1a2e')
    canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor='nw')

    #updates scroll area when the card list changes
    def on_frame_config(event):
        canvas.configure(scrollregion=canvas.bbox('all'))

    # makes sure that all cards fit in the window, if not window is resized
    def on_canvas_config(event):
        canvas.itemconfig(canvas_window, width=event.width)

    scrollable_frame.bind('<Configure>', on_frame_config)
    canvas.bind('<Configure>', on_canvas_config)

    #allows user to scroll on windows
    def on_mousewheel(event):
        canvas.yview_scroll(int(-1*(event.delta/120)), 'units')
    canvas.bind_all('<MouseWheel>', on_mousewheel)

    #allows users on mac to scroll
    def on_mousewheel_mac(event):
        canvas.yview_scroll(int(-1*event.delta),'units')
    canvas.bind_all('<Button-4>', on_mousewheel_mac)
    canvas.bind_all('<Button-5>', on_mousewheel_mac)

    # grabs new results and redraws the cards
    def refresh_cards(genre, min_rating):
        results = recommend_movies(1, 10, genre_filter=genre if genre else None,
                                   min_rating=min_rating)
        count_label.config(text=f"Showing {len(results)} recommendations")
        build_cards(scrollable_frame, results)
        canvas.yview_moveto(0)



    #loads the first set of recommendations when the app opens
    initial = recommend_movies(1, 10)
    count_label.config(text=f"Showing {len(initial)} recommendations")
    build_cards(scrollable_frame, initial)

    window.mainloop()

# opens window
show_ui()





from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
from models import Track, db
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import uuid  # For generating unique IDs if track_id is missing

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for flash messages

def generate_histogram(df):
    sns.histplot(data=df, x='album_release_year', bins=18)
    plt.xlabel('Year')
    plt.ylabel('Total Songs')
    hist_img = io.BytesIO()
    plt.savefig(hist_img, format='png')
    hist_img.seek(0)
    hist_url = base64.b64encode(hist_img.getvalue()).decode()
    plt.close()
    return hist_url

def generate_bar_chart(df):
    avg_popularity = df.groupby('playlist_genre')['track_popularity'].mean().reset_index()
    avg_popularity = avg_popularity.sort_values('track_popularity', ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x='playlist_genre', y='track_popularity', data=avg_popularity, palette='viridis')
    plt.xlabel('Genre')
    plt.ylabel('Average Popularity')
    plt.xticks(rotation=45, ha='right')
    bar_img = io.BytesIO()
    plt.savefig(bar_img, format='png')
    bar_img.seek(0)
    bar_url = base64.b64encode(bar_img.getvalue()).decode()
    plt.close()
    return bar_url

def generate_scatter_plot(df, max_points=500):
    if len(df) > max_points:
        df = df.sample(max_points, random_state=42)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='danceability', y='energy', hue='playlist_genre', data=df, alpha=0.7)
    plt.xlabel('Danceability')
    plt.ylabel('Energy')
    plt.legend(title='Genre', bbox_to_anchor=(1.05, 1), loc='upper left')
    scatter_img = io.BytesIO()
    plt.savefig(scatter_img, format='png', bbox_inches='tight')
    scatter_img.seek(0)
    scatter_url = base64.b64encode(scatter_img.getvalue()).decode()
    plt.close()
    return scatter_url

def generate_line_plot(df):
    df_grouped = df.groupby('album_release_year')['tempo'].mean().reset_index()
    sns.lineplot(x='album_release_year', y='tempo', data=df_grouped)
    plt.xlabel('Release Year')
    plt.ylabel('Average Tempo')
    plt.tight_layout()
    line_img = io.BytesIO()
    plt.savefig(line_img, format='png')
    line_img.seek(0)
    line_url = base64.b64encode(line_img.getvalue()).decode()
    plt.close()
    return line_url

def generate_pie_chart(df):
    plt.figure(figsize=(8, 8))
    category_counts = df['playlist_genre'].value_counts()
    category_counts.plot.pie(autopct='%1.1f%%', startangle=90, cmap='viridis')
    plt.ylabel('')  # Remove y-axis label
    pie_img = io.BytesIO()
    plt.savefig(pie_img, format='png')
    pie_img.seek(0)
    pie_url = base64.b64encode(pie_img.getvalue()).decode()
    plt.close()
    return pie_url

def generate_heatmap(df):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df[['danceability', 'energy', 'valence', 'tempo', 'track_popularity']].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    heatmap_img = io.BytesIO()
    plt.savefig(heatmap_img, format='png')
    heatmap_img.seek(0)
    heatmap_url = base64.b64encode(heatmap_img.getvalue()).decode()
    plt.close()
    return heatmap_url

def generate_plot(df):
    if df.empty:
        return None, None, None, None

    df['album_release_year'] = pd.DatetimeIndex(df['track_album_release_date']).year
    hist_url = generate_histogram(df)
    bar_url = generate_bar_chart(df)
    scatter_url = generate_scatter_plot(df)
    line_url = generate_line_plot(df)
    return hist_url, bar_url, scatter_url, line_url

# Helper function to validate and clean the DataFrame
def clean_dataframe(df, required_columns):
    # Ensure all required columns are present
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        raise ValueError(f"Missing required columns: {', '.join(missing_cols)}")

    # Fill missing or invalid `track_id` with unique IDs
    df['track_id'] = df['track_id'].fillna('').apply(lambda x: x if x else str(uuid.uuid4()))

    # Handle missing or invalid `duration_ms` values
    if df['duration_ms'].isnull().any() or not pd.api.types.is_numeric_dtype(df['duration_ms']):
        df['duration_ms'] = df['duration_ms'].fillna(0).astype(int)

    # Sanitize text fields to remove problematic characters
    text_fields = ['track_name', 'track_artist', 'track_album_name', 'playlist_name', 'playlist_genre', 'playlist_subgenre']
    for field in text_fields:
        if field in df.columns:
            df[field] = df[field].fillna('').apply(lambda x: x.replace('$', '').strip())

    # Ensure all other required columns are filled
    for col in required_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna('Unknown')  # Replace missing values with 'Unknown'

    return df

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if not file:
            flash("No file uploaded. Please upload a valid CSV file.", "error")
            return redirect(request.url)

        try:
            df = pd.read_csv(file)
        except Exception as e:
            flash(f"Error reading file: {e}", "error")
            return redirect(request.url)

        required_columns = [
            'track_id', 'track_name', 'track_artist', 'track_popularity',
            'danceability', 'energy', 'playlist_genre', 'track_album_release_date',
            'tempo', 'duration_ms'
        ]

        try:
            # Clean and validate the DataFrame
            df = clean_dataframe(df, required_columns)

            # Save data to the database
            with db.atomic():
                for _, row in df.iterrows():
                    track_data = {col: row[col] for col in required_columns}
                    # Use get_or_create to avoid duplicate track_id errors
                    Track.get_or_create(track_id=track_data['track_id'], defaults=track_data)

            flash("File uploaded and data saved successfully!", "success")
        except ValueError as ve:
            flash(str(ve), "error")
            return redirect(request.url)
        except Exception as e:
            flash(f"Error saving data to the database: {e}", "error")
            return redirect(request.url)

        # Redirect to visualizations page after successful upload
        return redirect(url_for('visualizations'))

    return render_template('upload.html')

@app.route('/visualizations', methods=['GET', 'POST'])
def visualizations():
    query = Track.select().dicts()
    df = pd.DataFrame(query)

    if df.empty:
        flash("No data available for visualization. Please upload data first.", "error")
        return redirect(url_for('upload_file'))

    # Ensure 'album_release_year' is created from 'track_album_release_date'
    if 'track_album_release_date' in df.columns:
        df['album_release_year'] = pd.to_datetime(df['track_album_release_date'], errors='coerce').dt.year
    else:
        flash("The dataset is missing the 'track_album_release_date' column.", "error")
        return redirect(url_for('upload_file'))

    # Get user inputs for filters
    min_popularity = request.form.get('min_popularity', type=int) or 0
    selected_genre = request.form.get('playlist_genre')
    max_points = request.form.get('max_points', type=int) or 500  # Default to 500 points
    year_start = request.form.get('year_start', type=int)
    year_end = request.form.get('year_end', type=int)

    # Apply filters
    if selected_genre:
        df = df[df['playlist_genre'] == selected_genre]
    df = df[df['track_popularity'] >= min_popularity]
    if year_start:
        df = df[df['album_release_year'] >= year_start]
    if year_end:
        df = df[df['album_release_year'] <= year_end]

    if df.empty:
        flash("No data available after applying filters. Please adjust the filters.", "error")
        return redirect(request.url)

    # Generate visualizations
    pie_url = generate_pie_chart(df)  # Fixed category: playlist_genre
    heatmap_url = generate_heatmap(df)
    hist_url, bar_url, scatter_url, line_url = generate_plot(df)

    # Generate scatter plot with all genres and limited points
    scatter_url = generate_scatter_plot(df, max_points=max_points)

    genres = df['playlist_genre'].unique() if not df.empty else []

    return render_template('visualizations.html', hist_url=hist_url, bar_url=bar_url,
                           scatter_url=scatter_url, line_url=line_url, pie_url=pie_url,
                           heatmap_url=heatmap_url, genres=genres)

@app.route('/reset', methods=['POST'])
def reset_data():
    try:
        Track.delete().execute()  # Delete all records from the Track table
        flash("All data has been reset successfully.", "success")
    except Exception as e:
        flash(f"Error resetting data: {e}", "error")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)

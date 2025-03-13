<template>
  <div class="max-w-3xl mx-auto p-4 sm:p-6">
    <!-- Header Section -->
    <div class="text-center mb-8">
      <h1 class="text-3xl font-bold text-gray-800 mb-2">Movie Prediction</h1>
      <p class="text-gray-600">
        Enter a movie summary and we'll predict the genre
      </p>
    </div>

    <!-- Form Section -->
    <div class="mb-6">
      <textarea
        v-model="summary"
        placeholder="Enter your movie summary here..."
        rows="6"
        class="w-full p-4 border border-gray-300 rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-y transition"
        :disabled="loading"
        :class="{ 'bg-gray-100': loading }"
      ></textarea>
      <button
        @click="predictMovie"
        :disabled="loading || !summary.trim()"
        class="mt-4 w-full sm:w-auto px-6 py-3 bg-blue-600 text-white font-medium rounded-lg shadow-sm hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition"
      >
        <span v-if="loading" class="flex items-center justify-center">
          <svg
            class="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
            xmlns="http://www.w3.org/2000/svg"
            fill="none"
            viewBox="0 0 24 24"
          >
            <circle
              class="opacity-25"
              cx="12"
              cy="12"
              r="10"
              stroke="currentColor"
              stroke-width="4"
            ></circle>
            <path
              class="opacity-75"
              fill="currentColor"
              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
            ></path>
          </svg>
          Predicting...
        </span>
        <span v-else>Predict Movie</span>
      </button>
    </div>

    <!-- Loading State -->
    <div v-if="loading" class="flex flex-col items-center justify-center py-8">
      <div
        class="w-16 h-16 border-4 border-blue-200 border-t-blue-600 rounded-full animate-spin"
      ></div>
      <p class="mt-4 text-gray-600">Analyzing your movie summary...</p>
    </div>

    <!-- Results Section -->
    <div
      v-if="prediction && !loading"
      class="mt-8 bg-white rounded-lg shadow-md overflow-hidden"
    >
      <div class="bg-blue-600 p-4">
        <h2 class="text-xl font-semibold text-white">Similar Movies</h2>
      </div>
      <div class="p-6">
        <div class="space-y-4">
          <div
            v-if="prediction.similar_movies && prediction.similar_movies.length"
            class="mt-6"
          >
            <div class="space-y-4">
              <div
                v-for="(movie, index) in prediction.similar_movies"
                :key="index"
                class="bg-gray-50 rounded-lg p-4 border border-gray-100 hover:shadow-md transition-shadow"
              >
                <div class="flex justify-between items-start">
                  <h4 class="text-md font-semibold text-gray-800">
                    {{ movie.movie_name }} ({{ movie.year }})
                  </h4>
                  <span
                    class="bg-blue-100 text-blue-800 text-xs font-medium px-2.5 py-0.5 rounded"
                  >
                    {{ (movie.similarity_score * 100).toFixed(1) }}% match
                  </span>
                </div>

                <div class="mt-2">
                  <p class="text-sm text-gray-600">
                    Genres: {{ formatGenres(movie.genres) }}
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- Error Message -->
    <div
      v-if="error"
      class="mt-6 bg-red-50 border border-red-200 rounded-lg p-4"
    >
      <div class="flex">
        <svg
          class="h-5 w-5 text-red-500 mr-3"
          xmlns="http://www.w3.org/2000/svg"
          viewBox="0 0 20 20"
          fill="currentColor"
        >
          <path
            fill-rule="evenodd"
            d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zm-1 9a1 1 0 01-1-1v-4a1 1 0 112 0v4a1 1 0 01-1 1z"
            clip-rule="evenodd"
          />
        </svg>
        <p class="text-red-700">{{ error }}</p>
      </div>
    </div>
  </div>
</template>

<script>
export default {
  name: "HelloWorld",
  data() {
    return {
      summary: "",
      prediction: null,
      loading: false,
      error: null,
    };
  },
  methods: {
    async predictMovie() {
      this.loading = true;
      this.error = null;
      this.prediction = null;

      try {
        const response = await fetch("http://127.0.0.1:5001/predict", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ summary: this.summary }),
          mode: "cors",
          credentials: "omit",
        });

        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }

        const data = await response.json();
        console.log("API response:", data);

        if (data.predictions && Array.isArray(data.predictions)) {
          this.prediction = {
            genre: this.extractMainGenre(data.predictions),
            confidence: data.predictions[0]?.similarity_score || 0.5,
            similar_movies: data.predictions,
          };
        } else {
          // Handle fallback case
          this.prediction = data;
        }

        console.log("Processed prediction:", this.prediction);
      } catch (error) {
        console.error("Error predicting movie:", error);
        if (error.message.includes("Failed to fetch")) {
          this.error =
            "Cannot connect to prediction service. Please ensure the backend server is running.";
        } else {
          this.error = `Prediction failed: ${error.message}`;
        }
      } finally {
        this.loading = false;
      }
    },

    // Extract the main genre from predictions
    extractMainGenre(predictions) {
      if (!predictions || !predictions.length) return "Unknown";

      try {
        // Get the top prediction's genres
        const topGenres = JSON.parse(predictions[0].genres.replace(/'/g, '"'));
        return topGenres[0].name || "Unknown";
      } catch (e) {
        console.error("Error parsing genres:", e);
        return "Unknown";
      }
    },

    // Format the genres string into readable text
    formatGenres(genresStr) {
      try {
        const genres = JSON.parse(genresStr.replace(/'/g, '"'));
        return genres.map((g) => g.name).join(", ");
      } catch (e) {
        console.error("Error formatting genres:", e);
        return genresStr;
      }
    },
  },
};
</script>

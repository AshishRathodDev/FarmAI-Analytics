
const API_BASE = "http://localhost:5050/api";

// POST image to Flask predict API
export const predictDisease = async (imageFile) => {
  try {
    const formData = new FormData();
    formData.append('file', imageFile);

    const response = await fetch(`${API_BASE}/predict`, {
      method: "POST",
      body: formData,
    });

    const data = await response.json();
    return data;

  } catch (error) {
    console.error("Error calling predict API:", error);
    return { status: "error", message: "Failed to connect to API" };
  }
};

// POST message to Flask chat API
export const chatWithAI = async (message) => {
  try {
    const response = await fetch(`${API_BASE}/chat`, {
      method: "POST",
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message }),
    });
    const data = await response.json();
    return data.response || "Sorry, could not get response.";
  } catch (error) {
    console.error("Error calling chat API:", error);
    return "Failed to connect to AI Chat API.";
  }
};

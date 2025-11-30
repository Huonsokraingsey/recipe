// Global variables
let selectedImages = [];
let selectedMeat = [];

// DOM elements
const imageUpload = document.getElementById("imageUpload");
const imagePreview = document.getElementById("imagePreview");
const detectedOutput = document.getElementById("detectedOutput");
const recommendBtn = document.getElementById("recommendBtn");

// Initialize when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    initializeEventListeners();
    // Auto-select "None" on page load
    const noneCard = document.querySelector('.meat-card[data-meat="None"]');
    if (noneCard) {
        noneCard.classList.add("selected");
        noneCard.setAttribute('aria-pressed', 'true');
    }
});

function initializeEventListeners() {
    // Image upload and preview
    if (imageUpload) {
        imageUpload.addEventListener("change", handleImageUpload);
    }

    // Meat selection
    initializeMeatSelection();

    // Recommendation button
    if (recommendBtn) {
        recommendBtn.addEventListener("click", handleRecommendation);
    }
}

// Handle image upload and processing
function handleImageUpload() {
    selectedImages = Array.from(imageUpload.files);
    imagePreview.innerHTML = "";

    if (!selectedImages.length) {
        imagePreview.textContent = "No image selected";
        detectedOutput.textContent = "—";
        return;
    }

    // Create image previews
    selectedImages.forEach(file => {
        const img = document.createElement("img");
        img.src = URL.createObjectURL(file);
        img.alt = "Uploaded ingredient";
        imagePreview.appendChild(img);
    });

    // Show loading state
    detectedOutput.textContent = "Detecting ingredients...";

    // Send to server for processing
    const formData = new FormData();
    selectedImages.forEach(img => formData.append("images", img));

    fetch("/upload", {
        method: "POST",
        body: formData
    })
    .then(res => {
        if (!res.ok) {
            throw new Error('Network response was not ok');
        }
        return res.json();
    })
    .then(data => {
        detectedOutput.textContent = data.detectedIngredients.join(", ");
    })
    .catch(error => {
        console.error('Error uploading images:', error);
        detectedOutput.textContent = "Error detecting ingredients";
    });
}

// Initialize meat selection functionality
function initializeMeatSelection() {
    const meatCards = document.querySelectorAll(".meat-card");
    
    meatCards.forEach(card => {
        card.addEventListener("click", handleMeatSelection);
        
        // Add keyboard accessibility
        card.addEventListener("keydown", (e) => {
            if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                handleMeatSelection.call(card, e);
            }
        });
        
        // Set tabindex for accessibility
        card.setAttribute('tabindex', '0');
        card.setAttribute('role', 'button');
        card.setAttribute('aria-pressed', 'false');
    });
}

// Handle meat selection with enhanced visual feedback
function handleMeatSelection(event) {
    const meat = this.dataset.meat;
    const isSelected = this.classList.contains("selected");

    if (meat === "None") {
        // Only clear other selections if "None" wasn't already selected
        if (!isSelected) {
            clearAllMeatSelections();
            this.classList.add("selected");
            this.setAttribute('aria-pressed', 'true');
            selectedMeat = [];
        }
        // If "None" was already selected, do nothing (keep it selected)
        return;
    }

    // Remove "None" selection if another meat is selected
    const noneCard = document.querySelector('.meat-card[data-meat="None"]');
    if (noneCard && noneCard.classList.contains("selected")) {
        noneCard.classList.remove("selected");
        noneCard.setAttribute('aria-pressed', 'false');
    }


    // Toggle current selection
    this.classList.toggle("selected");
    this.setAttribute('aria-pressed', this.classList.contains("selected").toString());

    // Update selectedMeat array
    if (isSelected) {
        selectedMeat = selectedMeat.filter(m => m !== meat);
    } else {
        selectedMeat.push(meat);
    }

    // If no meat selected after toggling, auto-select "None"
    if (selectedMeat.length === 0) {
        const noneCard = document.querySelector('.meat-card[data-meat="None"]');
        if (noneCard && !noneCard.classList.contains("selected")) {
            noneCard.classList.add("selected");
            noneCard.setAttribute('aria-pressed', 'true');
        }
    }
}

// Clear all meat selections
function clearAllMeatSelections() {
    document.querySelectorAll(".meat-card").forEach(card => {
        card.classList.remove("selected");
        card.setAttribute('aria-pressed', 'false');
    });
}

// Handle recipe recommendation
function handleRecommendation() {
    const ingredients = detectedOutput.textContent.split(", ").filter(Boolean);
    
    // Validation
    if (ingredients.length === 0 || ingredients[0] === "—" || ingredients[0] === "Detecting ingredients..." || ingredients[0] === "Error detecting ingredients") {
        alert("Please upload images of ingredients first!");
        return;
    }

    // Show loading state
    const recommendOutput = document.getElementById("recommendOutput");
    recommendBtn.disabled = true;
    recommendBtn.innerHTML = '<span class="loading"></span> Finding Recipes...';
    
    recommendOutput.innerHTML = `
        <div style="text-align: center; padding: 40px 20px;">
            <div class="loading" style="width: 50px; height: 50px; margin: 0 auto 20px;"></div>
            <h3 style="color: #ff7043; margin-bottom: 15px;">Finding Your Perfect Recipe</h3>
            <p>🔍 Analyzing your ingredients and finding the perfect recipe...</p>
        </div>
    `;

    // Send request to server
    fetch("/recommend", {
        method: "POST",
        headers: { 
            "Content-Type": "application/json" 
        },
        body: JSON.stringify({ 
            ingredients: ingredients, 
            meat: selectedMeat 
        })
    })
    .then(res => {
        if (!res.ok) {
            throw new Error('Network response was not ok');
        }
        return res.json();
    })
    .then(data => {
        displayRecommendation(data);
    })
    .catch(error => {
        console.error('Error getting recommendation:', error);
        recommendOutput.innerHTML = `
            <div class="no-results">
                <h3>❌ Error</h3>
                <p>Unable to get recommendation. Please try again.</p>
                <p><small>${error.message}</small></p>
            </div>
        `;
    })
    .finally(() => {
        recommendBtn.disabled = false;
        recommendBtn.textContent = "Show Recommendation";
    });
}

// Display the recommendation results
function displayRecommendation(data) {
    const recommendOutput = document.getElementById("recommendOutput");
    
    if (!data.recommendedDish || data.recommendedDish === "No matching recipes found.") {
        recommendOutput.innerHTML = `
            <div class="no-results">
                <h3>😞 No Recipe Found</h3>
                <p>We couldn't find a recipe matching your ingredients.</p>
                <p>Try uploading different ingredients or selecting different meat options.</p>
            </div>
        `;
        return;
    }


    // Format instructions with better spacing
    const formattedInstructions = formatInstructions(data.details.instructions);
    
    recommendOutput.innerHTML = `
        <h3>🍛 ${data.recommendedDish}</h3>
        ${data.image ? `
            <img src="${data.image}" alt="${data.recommendedDish}" class="food-image">
        ` : ''}
        
        <div style="margin-top: 25px;">
            <h4>📋 Ingredients</h4>
            <ul>
                ${data.details.ingredients.map(i => `<li>${i}</li>`).join("")}
            </ul>
        </div>
        
        <div style="margin-top: 25px;">
            <h4>👩‍🍳 Instructions</h4>
            <div class="recipe-instructions">
                <p>${formattedInstructions}</p>
            </div>
        </div>
        
        <div class="ingredients-summary">
            <p><strong>🎯 Based on your ingredients:</strong> ${data.detected.join(", ")}</p>
        </div>
    `;
}

// Format instructions with better spacing and line breaks
function formatInstructions(instructions) {
    if (!instructions) return 'No instructions available.';
    
    // Replace numbered steps with line breaks
    let formatted = instructions
        .replace(/(\d+\.)/g, '\n$1')  // Add line break before each numbered step
        .replace(/\n\s*\n/g, '\n')    // Remove multiple line breaks
        .trim();
    
    // If no numbers found, try to split by periods
    if (!formatted.includes('\n')) {
        formatted = instructions
            .split('.')
            .filter(step => step.trim().length > 0)
            .map(step => step.trim() + '.')
            .join('\n\n');
    }
    
    return formatted;
}

// Utility function to reset the form
function resetForm() {
    selectedImages = [];
    selectedMeat = [];
    
    if (imageUpload) imageUpload.value = '';
    if (imagePreview) imagePreview.innerHTML = 'No image selected';
    if (detectedOutput) detectedOutput.textContent = '—';
    
    clearAllMeatSelections();
    const noneCard = document.querySelector('.meat-card[data-meat="None"]');
    if (noneCard) {
        noneCard.classList.add("selected");
        noneCard.setAttribute('aria-pressed', 'true');
    }
    
    const recommendOutput = document.getElementById("recommendOutput");
    if (recommendOutput) {
        recommendOutput.innerHTML = '<p>Your food recommendation will appear here...</p>';
    }
}

// Add reset functionality if needed
document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
        resetForm();
    }
});

// Add click outside to deselect (optional)
document.addEventListener('click', function(e) {
    if (!e.target.closest('.meat-card') && !e.target.closest('.meat-options')) {
        // Keep current selection, don't auto-deselect
    }
});

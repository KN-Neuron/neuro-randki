// Neuro-Randki — Dark Mode Toggle
document.addEventListener('DOMContentLoaded', () => {
    const toggleBtn = document.getElementById('dark-mode-toggle');
    if (!toggleBtn) return;

    function updateToggle() {
        const isDark = localStorage.getItem('darkMode') === 'true';
        toggleBtn.textContent = isDark ? 'TRYB UROCZY' : 'TRYB MROCZNY';
        
        // Swap heart icon
        const heartImg = document.querySelector('.mode-toggle-heart');
        if (heartImg) {
            if (isDark) {
                heartImg.src = heartImg.src.replace('heart_red.svg', 'heart_black.svg');
            } else {
                heartImg.src = heartImg.src.replace('heart_black.svg', 'heart_red.svg');
            }
        }
    }

    toggleBtn.addEventListener('click', (e) => {
        e.preventDefault();
        const isDark = localStorage.getItem('darkMode') === 'true';
        const newVal = !isDark;
        localStorage.setItem('darkMode', newVal);
        document.documentElement.classList.toggle('dark-mode', newVal);
        updateToggle();
    });

    updateToggle();
});

console.log("Neuro-Randki initialized.");

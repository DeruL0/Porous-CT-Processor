/**
 * Presentation Controller
 * Handles slide navigation via keyboard.
 */

document.addEventListener('DOMContentLoaded', () => {
    const slides = document.querySelectorAll('.slide');
    let currentSlide = 0;

    const showSlide = (index) => {
        // Bounds check
        if (index < 0) index = 0;
        if (index >= slides.length) index = slides.length - 1;

        // Update classes
        slides.forEach((slide, i) => {
            slide.classList.remove('active');
            if (i === index) {
                slide.classList.add('active');
            }
        });

        currentSlide = index;
    };

    // Keyboard Navigation
    document.addEventListener('keydown', (e) => {
        switch (e.key) {
            case 'ArrowRight':
            case ' ':
            case 'PageDown':
                showSlide(currentSlide + 1);
                break;
            case 'ArrowLeft':
            case 'PageUp':
                showSlide(currentSlide - 1);
                break;
        }
    });

    console.log("Presentation Mode Loaded: " + slides.length + " slides.");
});

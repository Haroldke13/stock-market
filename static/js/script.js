// Add interactivity to the app
document.addEventListener('DOMContentLoaded', function () {
    const form = document.querySelector('form');

    form.addEventListener('submit', function (e) {
        const inputField = document.querySelector('input[type="text"], select');
        if (!inputField.value.trim()) {
            e.preventDefault();
            alert("Please enter or select a valid stock symbol!");
        }
    });

    // Example: Highlight form on focus
    const inputField = document.querySelector('input[type="text"], select');
    inputField.addEventListener('focus', function () {
        inputField.style.border = "2px solid #007bff";
    });

    inputField.addEventListener('blur', function () {
        inputField.style.border = "1px solid #ddd";
    });
});

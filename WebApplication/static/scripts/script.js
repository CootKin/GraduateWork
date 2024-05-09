function checkInput() {
    var login_input = document.getElementById("login-input");
    var password_input = document.getElementById("password-input");
    var login_star = document.getElementById("login-star");
    var password_star = document.getElementById("password-star");

    if (login_input.value.length > 0) {
        login_star.style.display = "none";
    } else {
        login_star.style.display = "inline";
    }

    if (password_input.value.length > 0) {
        password_star.style.display = "none";
    } else {
        password_star.style.display = "inline";
    }
}

function showAlert(message) {
    alert(message);
}
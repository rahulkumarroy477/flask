function openPopup() {
    popup.style.display = "block";
    document.getElementById("btn").style.display = "none";
}

function closePopup() {  
    document.getElementById("popup").style.display = "none";
    document.getElementById("btn").style.display = "block";
    location.reload(true)
}
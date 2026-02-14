window.HELP_IMPROVE_VIDEOJS = false;

var INTERP_BASE = "./static/interpolation/stacked";
var NUM_INTERP_FRAMES = 240;

var interp_images = [];
function preloadInterpolationImages() {
  for (var i = 0; i < NUM_INTERP_FRAMES; i++) {
    var path = INTERP_BASE + '/' + String(i).padStart(6, '0') + '.jpg';
    interp_images[i] = new Image();
    interp_images[i].src = path;
  }
}

function setInterpolationImage(i) {
  var image = interp_images[i];
  image.ondragstart = function () { return false; };
  image.oncontextmenu = function () { return false; };
  $('#interpolation-image-wrapper').empty().append(image);
}

// Draw dashed connection lines between boxes
function drawConnectionLines() {
  // Method Overview: left box to right box
  drawLineBetweenBoxes('.method-description-box', '.method-image-container', 'method-connection-svg');

  // Complete Pipeline: left box to right box
  drawLineBetweenBoxes('.pipeline-image-box', '.pipeline-description-box', 'pipeline-connection-svg');
}

function drawLineBetweenBoxes(leftSelector, rightSelector, svgId) {
  var leftBox = document.querySelector(leftSelector);
  var rightBox = document.querySelector(rightSelector);

  if (!leftBox || !rightBox) return;

  // Remove existing SVG if any
  var existingSvg = document.getElementById(svgId);
  if (existingSvg) {
    existingSvg.remove();
  }

  var leftRect = leftBox.getBoundingClientRect();
  var rightRect = rightBox.getBoundingClientRect();

  // Get the parent container position for absolute positioning
  var container = leftBox.closest('.columns');
  if (!container) return;

  var containerRect = container.getBoundingClientRect();

  // Calculate positions relative to container
  var leftRight = leftRect.right - containerRect.left;
  var rightLeft = rightRect.left - containerRect.left;
  var leftTop = leftRect.top - containerRect.top;
  var rightTop = rightRect.top - containerRect.top;
  var leftBottom = leftRect.bottom - containerRect.top;
  var rightBottom = rightRect.bottom - containerRect.top;

  // Create SVG element
  var svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.id = svgId;
  svg.style.position = 'absolute';
  svg.style.left = '0';
  svg.style.top = '0';
  svg.style.width = '100%';
  svg.style.height = '100%';
  svg.style.pointerEvents = 'none';
  svg.style.zIndex = '1';

  // Draw top line
  var topLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  topLine.setAttribute('x1', leftRight);
  topLine.setAttribute('y1', leftTop + 20); // Offset from top
  topLine.setAttribute('x2', rightLeft);
  topLine.setAttribute('y2', rightTop + 20);
  topLine.setAttribute('stroke', '#808080');
  topLine.setAttribute('stroke-width', '2');
  topLine.setAttribute('stroke-dasharray', '8,4');
  topLine.setAttribute('opacity', '0.6');

  // Draw bottom line
  var bottomLine = document.createElementNS('http://www.w3.org/2000/svg', 'line');
  bottomLine.setAttribute('x1', leftRight);
  bottomLine.setAttribute('y1', leftBottom - 20); // Offset from bottom
  bottomLine.setAttribute('x2', rightLeft);
  bottomLine.setAttribute('y2', rightBottom - 20);
  bottomLine.setAttribute('stroke', '#808080');
  bottomLine.setAttribute('stroke-width', '2');
  bottomLine.setAttribute('stroke-dasharray', '8,4');
  bottomLine.setAttribute('opacity', '0.6');

  svg.appendChild(topLine);
  svg.appendChild(bottomLine);

  // Make container position relative
  container.style.position = 'relative';
  container.appendChild(svg);
}

// Redraw lines on window resize
function handleResize() {
  drawConnectionLines();
}

$(document).ready(function () {
  // Check for click events on the navbar burger icon
  $(".navbar-burger").click(function () {
    // Toggle the "is-active" class on both the "navbar-burger" and the "navbar-menu"
    $(".navbar-burger").toggleClass("is-active");
    $(".navbar-menu").toggleClass("is-active");

  });

  var options = {
    slidesToScroll: 1,
    slidesToShow: 3,
    loop: true,
    infinite: true,
    autoplay: false,
    autoplaySpeed: 3000,
  }

  // Initialize all div with carousel class
  var carousels = bulmaCarousel.attach('.carousel', options);

  // Loop on each carousel initialized
  for (var i = 0; i < carousels.length; i++) {
    // Add listener to  event
    carousels[i].on('before:show', state => {
      console.log(state);
    });
  }

  // Access to bulmaCarousel instance of an element
  var element = document.querySelector('#my-element');
  if (element && element.bulmaCarousel) {
    // bulmaCarousel instance is available as element.bulmaCarousel
    element.bulmaCarousel.on('before-show', function (state) {
      console.log(state);
    });
  }

  /*var player = document.getElementById('interpolation-video');
  player.addEventListener('loadedmetadata', function() {
    $('#interpolation-slider').on('input', function(event) {
      console.log(this.value, player.duration);
      player.currentTime = player.duration / 100 * this.value;
    })
  }, false);*/
  preloadInterpolationImages();

  $('#interpolation-slider').on('input', function (event) {
    setInterpolationImage(this.value);
  });
  setInterpolationImage(0);
  $('#interpolation-slider').prop('max', NUM_INTERP_FRAMES - 1);

  bulmaSlider.attach();

  // Draw connection lines after page load
  setTimeout(function () {
    drawConnectionLines();
  }, 100);

  // Redraw on window resize
  var resizeTimer;
  $(window).on('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      drawConnectionLines();
    }, 250);
  });

  // Image modal functionality
  initImageModal();

})

// Image modal for click-to-enlarge
function initImageModal() {
  // Create modal HTML if it doesn't exist
  if (!document.getElementById('image-modal')) {
    var modalHTML = `
      <div id="image-modal" class="image-modal">
        <button class="image-modal-close">&times;</button>
        <img class="image-modal-content" id="modal-img">
      </div>
    `;
    document.body.insertAdjacentHTML('beforeend', modalHTML);
  }

  var modal = document.getElementById('image-modal');
  var modalImg = document.getElementById('modal-img');
  var closeBtn = document.querySelector('.image-modal-close');

  // Function to add click handler to images
  function addImageClickHandler(selector) {
    var images = document.querySelectorAll(selector);
    images.forEach(function (img) {
      img.addEventListener('click', function () {
        modal.classList.add('active');
        modalImg.src = this.src;
        document.body.style.overflow = 'hidden'; // Prevent scrolling
      });
    });
  }

  // Add click event to pipeline images
  addImageClickHandler('.pipeline-image-box img');

  // Add click event to method images
  addImageClickHandler('.method-image-container img');

  // Add click event to experimental results images
  addImageClickHandler('.content img');

  // Close modal on close button click
  closeBtn.addEventListener('click', closeModal);

  // Close modal on background click
  modal.addEventListener('click', function (e) {
    if (e.target === modal) {
      closeModal();
    }
  });

  // Close modal on ESC key
  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && modal.classList.contains('active')) {
      closeModal();
    }
  });

  function closeModal() {
    modal.classList.remove('active');
    document.body.style.overflow = ''; // Restore scrolling
  }
}

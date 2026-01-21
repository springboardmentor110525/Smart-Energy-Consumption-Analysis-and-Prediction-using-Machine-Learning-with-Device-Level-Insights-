document.addEventListener('DOMContentLoaded', () => {
    const sidebar = document.querySelector('.sidebar');
    const mainWrapper = document.querySelector('.main-wrapper');
    const toggleBtn = document.querySelector('.nav-toggle'); // Mobile toggle
    const overlay = document.querySelector('.nav-overlay');
    const desktopToggle = document.querySelector('.sidebar-toggle-desktop'); // Desktop toggle

    // Check saved state
    const isCollapsed = localStorage.getItem('sidebarCollapsed') === 'true';
    if (isCollapsed && window.innerWidth > 768) {
        sidebar.classList.add('collapsed');
        mainWrapper.classList.add('collapsed');
    }

    // Mobile Toggle
    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            sidebar.classList.toggle('open');
            overlay.classList.toggle('show');
        });
    }

    if (overlay) {
        overlay.addEventListener('click', () => {
            sidebar.classList.remove('open');
            overlay.classList.remove('show');
        });
    }

    // Desktop Collapse Toggle
    if (desktopToggle) {
        desktopToggle.addEventListener('click', () => {
            sidebar.classList.toggle('collapsed');
            mainWrapper.classList.toggle('collapsed');

            // Save state
            const collapsed = sidebar.classList.contains('collapsed');
            localStorage.setItem('sidebarCollapsed', collapsed);
        });
    }

    // Handle window resize
    window.addEventListener('resize', () => {
        if (window.innerWidth <= 768) {
            sidebar.classList.remove('collapsed');
            mainWrapper.classList.remove('collapsed');
        } else {
            // Restore state on desktop
            const savedState = localStorage.getItem('sidebarCollapsed') === 'true';
            if (savedState) {
                sidebar.classList.add('collapsed');
                mainWrapper.classList.add('collapsed');
            }
        }
    });
});

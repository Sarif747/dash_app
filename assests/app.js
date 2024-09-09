document.addEventListener('DOMContentLoaded', function() {
    console.log("Custom JavaScript loaded");

    // Example: Add an event listener to dropdowns for additional interactivity
    const dropdowns = document.querySelectorAll('#x-dropdown, #y-dropdown, #chart-type-dropdown');
    
    dropdowns.forEach(dropdown => {
        dropdown.addEventListener('change', function() {
            // Log changes to the console
            console.log(`${this.id} changed to ${this.value}`);
            
            // Update chart title dynamically based on dropdown changes
            const chartTitle = document.getElementById('chart-title');
            if (chartTitle) {
                const xAxisValue = document.getElementById('x-dropdown').value;
                const yAxisValue = document.getElementById('y-dropdown').value;
                const chartType = document.getElementById('chart-type-dropdown').value;
                chartTitle.textContent = `${chartType.charAt(0).toUpperCase() + chartType.slice(1)} of ${xAxisValue}${yAxisValue ? ' vs ' + yAxisValue : ''}`;
            }
        });
    });
});
alert('If you see this alert, then your custom JavaScript script has run!')
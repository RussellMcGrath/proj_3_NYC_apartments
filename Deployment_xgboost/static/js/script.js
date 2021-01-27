// store buttons objects as a variable
var feature_button = d3.select("#buttons").selectAll("button");
// store form as variable
var form = d3.select("form");

var yesno_features = ["no_fee","has_roofdeck","has_washer_dryer","has_elevator","has_dishwasher","has_patio","has_gym","has_doorman"];
var yesno = ["Yes","No"];
var dropdownfields = ["borough", "neighborhood", "submarket"];
// store dropdown list values
var boroughsList = boroughs.list;
var neighborhoodsList = neighborhoods.list;
var submarketsList = submarkets.list;

// on button click...
feature_button.on("click", function() {
    // store clicked button as variable
    var selection = d3.select(this);
    // make button disappear
    selection.attr("style","visibility:hidden");
    // add label to form
    form.append("label")
        .text(selection.text()+":");
    // add input/selection field to form
    // if it is yes/no field...
    if(yesno_features.includes(selection.attr("name"))) {
        // add select field
        form.append("select")
            .attr("name",selection.attr("name"))
            .attr("class",selection.attr("name"))
            .attr("required","required")
            .append("option")
                .attr("selected","true")
                .attr("disabled","disabled")
                .text("-- select Yes/No --");
        // add yes/no options to the dropdown
        for (i=0;i<2;i++) {
            d3.select(`.${selection.attr("name")}`)
                .append("option")
                .attr("value",i)
                .text(yesno[i]);
        }
        // var select = d3.selectAll("select").select(`".${selection.attr("name")}"`);
        // for (i=0;i<2;i++) {
        //     select.append("option")
        //         .attr("value",i)
        //         .text(yesno[i]);
        // }
        form.append("br");
    // if it is a dropdown field
    } else if (dropdownfields.includes(selection.attr("name"))) {
        // determine which dropdown field has been added, and set the appropriate list of options
        var optionlist;
        if (selection.attr("name") === "borough") {
            optionlist = boroughsList;
        } else if (selection.attr("name") === "neighborhood") {
            optionlist = neighborhoodsList;
        } else if (selection.attr("name") === "submarket") {
            optionlist = submarketsList;
        }
        // add select field
        form.append("select")
            .attr("name",selection.attr("name"))
            .attr("class",selection.attr("name"))
            .attr("required","required")
            .append("option")
                .attr("selected","true")
                .attr("disabled","disabled")
                .text(`-- select ${selection.attr("name")} --`);
        // add dropdown options
        for (i=0;i<optionlist.length;i++) {
            d3.select(`.${selection.attr("name")}`)
                .append("option")
                .attr("value", optionlist[i])
                .text(optionlist[i]);
        }
        form.append("br");        
    // if it is an input field
    } else {        
        form.append("input")
            .attr("name",selection.attr("name"))
            .attr("required","required")
            .attr("placeholder","--enter value--");
        form.append("br");
    }
})
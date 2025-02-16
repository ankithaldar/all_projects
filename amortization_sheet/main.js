function amortization_calculator() {
  output_schedule_to_sheet(
    generate_amortization_schedule(
      capture_all_event_dates(
        get_tenure(
          read_and_validate_inputs()
        )
      )
    ), 'emi_calculator!I17')
}
